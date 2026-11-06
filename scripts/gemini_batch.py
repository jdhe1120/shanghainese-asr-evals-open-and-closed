"""A generateContent Gemini model over the Gemini Batch API (half price, no per-day request cap), into cache/<model>/.
Copied from the Cantonese evals' scripts/gemini_batch.py.

  uv run python scripts/gemini_batch.py MODEL submit [--limit N] [--clips a,b] [--set wubench]
  uv run python scripts/gemini_batch.py MODEL status      # job in cache/<NAME>_batch.json
  uv run python scripts/gemini_batch.py MODEL collect     # download results, write the cache

MODEL is a registry name whose module uses _gemini (gemini-argon). Same request as the live adapter (its build_request
and parse), so batch and live results are the same row. Only clips with no cached result are sent. Key: GEMINI_API (or
GEMINI_API_KEY); GEMINI_USE=free uses GEMINI_API_FREE_KEY. Batch results have no per-clip timing, so the cache entry has
"seconds": 0.0 and "batch": <job name>; token usage per clip is kept too, for the real cost.
Afterwards: uv run python -m shaeval run --no-ui --models all --cached-only to rebuild the run.
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shaeval import cache  # noqa: E402
from shaeval.clips import load_clips  # noqa: E402
from shaeval.models import REGISTRY  # noqa: E402
from shaeval.models import _gemini as gc  # noqa: E402
from shaeval.models._env import env  # noqa: E402

API = gc.API
MODEL_NAME = a = STATE = None  # set in main: registry name (cache folder), its module, its job file


def _which() -> str:
    return "free" if os.environ.get("GEMINI_USE") == "free" else "paid"


def _key() -> str:
    e = env()
    k = e.get("GEMINI_API_FREE_KEY") if _which() == "free" else e.get("GEMINI_API") or e.get("GEMINI_API_KEY")
    if not k:
        sys.exit("no Gemini key in .env (GEMINI_API, or GEMINI_API_FREE_KEY with GEMINI_USE=free)")
    return k


def _req(method: str, url: str, body=None, headers=None, raw=False):
    h = {"x-goog-api-key": _key(), **(headers or {})}
    if isinstance(body, (dict, list)):
        body, h["Content-Type"] = json.dumps(body).encode(), "application/json"
    r = urllib.request.urlopen(urllib.request.Request(url, data=body, method=method, headers=h), timeout=600)
    return r if raw else json.load(r)


def submit(limit: int | None, only: list[str] | None, sets: list[str] | None) -> None:
    if STATE.exists() and json.loads(STATE.read_text()).get("collected") is not True:
        sys.exit(f"a job is already open in {STATE}; run status/collect first (or delete the file to abandon it)")
    todo = [c for c in load_clips() if c.aggregate and cache.get(MODEL_NAME, c.id, c.digest) is None
            and (not only or c.id in only) and (not sets or c.set in sets)][:limit]
    if not todo:
        sys.exit("nothing to send: every clip has a cached result")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for c in todo:
            f.write(json.dumps({"key": f"{c.id}|{c.digest}", "request": a.build_request(str(c.wav))}) + "\n")
        path = Path(f.name)
    size = path.stat().st_size
    print(f"{len(todo)} clips, {size / 1e6:.0f} MB JSONL; uploading", flush=True)
    start = _req("POST", f"{API}/upload/v1beta/files", {"file": {"display_name": f"shaeval-{MODEL_NAME}-input"}},
                 {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                  "X-Goog-Upload-Header-Content-Length": str(size),
                  "X-Goog-Upload-Header-Content-Type": "application/jsonl"}, raw=True)
    upload_url = start.headers["x-goog-upload-url"]
    with path.open("rb") as fh:
        up = json.load(urllib.request.urlopen(urllib.request.Request(
            upload_url, data=fh, method="POST", headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                                         "X-Goog-Upload-Command": "upload, finalize"}), timeout=3600))
    path.unlink()
    file_name = up["file"]["name"]
    job = _req("POST", f"{API}/v1beta/models/{a.MODEL}:batchGenerateContent",
               {"batch": {"display_name": f"shaeval-{MODEL_NAME}", "input_config": {"file_name": file_name}}})
    STATE.write_text(json.dumps({"job": job["name"], "input_file": file_name, "clips": len(todo),
                                 "key": _which(), "collected": False}, indent=1))
    print(f"submitted {job['name']} ({len(todo)} clips); state in {STATE.relative_to(ROOT)}")


def _job() -> tuple[dict, dict]:
    st = json.loads(STATE.read_text())
    return st, _req("GET", f"{API}/v1beta/{st['job']}")


def _state_of(job: dict) -> str:
    return (job.get("metadata") or {}).get("state") or job.get("state") or ("DONE" if job.get("done") else "?")


def status() -> None:
    st, job = _job()
    meta = job.get("metadata") or {}
    print(st["job"], _state_of(job), json.dumps(meta.get("batchStats") or {}))
    if job.get("error"):
        print("error:", json.dumps(job["error"])[:500])


def collect() -> None:
    st, job = _job()
    state = _state_of(job)
    if "SUCCEEDED" not in state:
        sys.exit(f"job is {state}, not finished")
    out = ((job.get("response") or {}).get("responsesFile") or
           ((job.get("metadata") or {}).get("output") or {}).get("responsesFile"))
    if not out:
        sys.exit("job finished but no responsesFile: " + json.dumps(job)[:500])
    lines = _req("GET", f"{API}/download/v1beta/{out}:download?alt=media", raw=True).read().decode().splitlines()
    ok, failed, tokens = 0, [], {"in": 0, "audio": 0, "out": 0, "thoughts": 0}
    for line in filter(None, lines):
        r = json.loads(line)
        clip_id, digest = r["key"].split("|")
        resp = r.get("response") or {}
        cands = resp.get("candidates") or []
        if r.get("error") or not cands:
            failed.append((clip_id, json.dumps(r.get("error") or resp.get("promptFeedback") or "no candidates")[:120]))
            continue
        text = a.parse(gc.response_text(resp, a.MODEL))
        u = resp.get("usageMetadata") or {}
        tokens["in"] += u.get("promptTokenCount", 0)
        tokens["out"] += u.get("candidatesTokenCount", 0)
        tokens["thoughts"] += u.get("thoughtsTokenCount", 0)
        tokens["audio"] += sum(d.get("tokenCount", 0) for d in u.get("promptTokensDetails", []) if d.get("modality") == "AUDIO")
        cache.put(MODEL_NAME, clip_id, digest, {"text": text, "seconds": 0.0, "batch": st["job"], "usage": u})
        ok += 1
    st.update(collected=True, ok=ok, failed=failed, tokens=tokens)
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(f"cached {ok}, failed {len(failed)} (these can be re-run live or in a new batch)")
    for f in failed[:20]:
        print("  ", *f)
    print("tokens:", tokens)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", help="registry name, e.g. gemini-argon")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--limit", type=int)
    s.add_argument("--clips")
    s.add_argument("--set", action="append", dest="sets")
    sub.add_parser("status")
    sub.add_parser("collect")
    args = p.parse_args()
    if args.model not in REGISTRY:
        sys.exit(f"unknown model {args.model}")
    MODEL_NAME, a = args.model, importlib.import_module(f"shaeval.models.{REGISTRY[args.model]}")
    if not hasattr(a, "build_request"):
        sys.exit(f"{args.model} isn't a generateContent Gemini model (no build_request)")
    STATE = ROOT / "cache" / f"{a.NAME}_batch.json"
    if args.cmd == "submit":
        submit(args.limit, args.clips.split(",") if args.clips else None, args.sets)
    else:
        {"status": status, "collect": collect}[args.cmd]()
