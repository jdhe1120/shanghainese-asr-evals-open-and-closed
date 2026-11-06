"""Runs models over clips, caching hypotheses and writing progress as it goes.

Every cached result is filled in first; then the models still to transcribe form one queue: local models one at a
time (each unloaded before the next loads), hosted APIs in parallel threads beside them. runs/latest.json is rewritten
as cells finish (the UI polls it, at most about once a second), by this one run only: a second run while one is live
is refused, because two writers make the page flip between them. On finish the
same document is also saved as runs/<runId>.json and runs/<runId>.csv. Scores are computed here from cached text, so changing the
normalisation never needs a re-transcription.
"""
import csv
import json
import os
import threading
import time
import tomllib
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import cache, metrics
from .clips import Clip, clip_set_version, load_clips
from .models import REGISTRY, call_gap, in_column_order, describe, get_batcher, get_pacer, get_transcriber, is_local, unload_local, uses_cache

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _write(path: Path, doc: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # per process: parallel runs must not share a temp file
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# (fold interchangeable graphs, digits -> Chinese numerals); the UI picks one with its two toggles.
VARIANTS = {"strict": (False, False), "equiv": (True, False), "num": (False, True), "equivNum": (True, True)}


def _scored(clip: Clip, text: str) -> dict:
    scores = {}
    for name, (equiv, numerals) in VARIANTS.items():
        sc = metrics.score(clip.reference, text, equiv, numerals)
        scores[name] = {**sc, "diff": metrics.diff(clip.reference, text, equiv, numerals)}
    # cer / cerEquiv stay at the top level for the CSV and the app
    return {"scores": scores, "cer": scores["strict"]["cer"], "cerEquiv": scores["equiv"]["cer"],
            "refLen": scores["strict"]["refLen"]}


def _summary(doc: dict) -> dict:
    """model -> clip set -> scores; the sets are different kinds of speech, so they are never pooled."""
    out = {}
    clips = {c["id"]: c for c in doc["clips"]}
    sets = sorted({c["set"] for c in clips.values()})
    for model, cells in doc["cells"].items():
        out[model] = {}
        for s in sets:
            mine = [cid for cid, c in clips.items() if c["aggregate"] and c["set"] == s]
            done = [cells[cid] for cid in mine if cells[cid]["status"] == "done"]
            out[model][s] = {v: metrics.summarize([c["scores"][v] for c in done]) for v in VARIANTS}
            out[model][s].update(clipsDone=len(done), clipsTotal=len(mine))
    return out


FLAGS = Path(__file__).resolve().parent.parent / "rules" / "flags.toml"


def flags() -> list[dict]:
    """Results to read with care (rules/flags.toml), shown as markers on the page and in results/."""
    return tomllib.loads(FLAGS.read_text(encoding="utf-8")).get("flag", []) if FLAGS.exists() else []


def _refuse_if_running() -> None:
    """latest.json has one writer: if the run that wrote it is still alive, stop here rather than fight it."""
    try:
        live = json.loads((RUNS_DIR / "latest.json").read_text(encoding="utf-8"))
        if live.get("state") == "running" and live.get("pid"):
            os.kill(live["pid"], 0)
            raise SystemExit(f"run {live['runId']} is still going (pid {live['pid']}); wait for it, or stop it and start "
                             "one run that names every model you want: they share one queue.")
    except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError):
        pass


def run(models: list[str], only_clips: list[str] | None = None, force: bool = False,
        model_clips: dict[str, list[str]] | None = None, include_full: bool = False,
        cached_only: bool = False, only_sets: list[str] | None = None) -> dict:
    """model_clips limits one model to some clips (e.g. a paid API on a single clip); its other cells are "skipped".
    The whole-story *-full clips are always listed (shown in the UI, never averaged), but a model is only called on
    one with include_full; otherwise a cached result is shown or the cell is skipped."""
    model_clips = model_clips or {}
    _refuse_if_running()
    clips = load_clips()
    # --clips / --set choose what gets transcribed; every other clip is still listed with whatever is cached, so a
    # small run never hides the rest of the results from the UI.
    picked = {c.id for c in clips if (not only_clips or c.id in only_clips) and (not only_sets or c.set in only_sets)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Every other model with cached results is shown too (read-only, refreshed while this run goes), so a run of one
    # model never hides the others in the UI, even with several runs going in parallel.
    others = [m for m in REGISTRY if m not in models and uses_cache(m) and any((cache.CACHE_DIR / m).glob("*.json"))]
    doc = {
        "runId": run_id, "state": "running", "startedAt": run_id, "finishedAt": None,
        "clipSetVersion": clip_set_version(clips), "models": in_column_order(models + others), "runModels": models,
        "modelInfo": {m: describe(m) for m in models + others}, "current": [], "pid": os.getpid(), "flags": flags(),
        "normalization": "reference in the canonical spelling (rules/wu_canonical.toml), then NFKC, OpenCC t2s, punctuation/space stripped; 'equiv' also counts rules/wu_folds.toml 'same' pairs (and the corpus's own spelling) as matches",
        "clips": [{"id": c.id, "set": c.set, "seconds": round(c.seconds, 1), "reference": c.reference,
                   "aggregate": c.aggregate}
                  for c in clips],
        "cells": {m: {c.id: {"status": "pending"} for c in clips} for m in models + others},
    }
    latest = RUNS_DIR / "latest.json"

    def refresh_others() -> None:
        """Fill the other models' cells from their caches; only clips not yet filled are looked up and scored. Also
        adds any model whose first results were cached after this run started (e.g. by a parallel run)."""
        for m in REGISTRY:
            if m not in models and m not in others and uses_cache(m) and any((cache.CACHE_DIR / m).glob("*.json")):
                others.append(m)
                doc["models"] = in_column_order(doc["models"] + [m])
                doc["modelInfo"][m] = describe(m)
                doc["cells"][m] = {c.id: {"status": "pending"} for c in clips}
        for m in others:
            for c in clips:
                cell = doc["cells"][m][c.id]
                if cell["status"] == "done":
                    continue
                hit = cache.get(m, c.id, c.digest)
                if hit:
                    cell.clear()
                    cell.update(status="done", cached=True, text=hit["text"], seconds=round(hit["seconds"], 2),
                                rtf=round(hit["seconds"] / c.seconds, 2), **_scored(c, hit["text"]))
                else:
                    cell.update(status="skipped", note="not transcribed yet (another run may be doing it)")

    refresh_others()
    lock = threading.RLock()  # API models run in threads beside the local ones; every change to doc holds it
    last_flush = [0.0]

    def flush(force: bool = True) -> None:
        """Rewrite runs/latest.json, at most about once a second unless something the page shows live changed."""
        with lock:
            if not force and time.time() - last_flush[0] < 1.0:
                return
            last_flush[0] = time.time()
            doc["summary"] = _summary(doc)
            _write(latest, doc)

    # Fill every cached result before transcribing anything, so finished numbers are on the page from the start and
    # stay put; what's left forms the queue.
    todo: dict[str, list[Clip]] = {m: [] for m in models}
    for clip in clips:
        for model in models:
            cell = doc["cells"][model][clip.id]
            if model in model_clips and clip.id not in model_clips[model]:
                cell["status"] = "skipped"
                continue
            hit = None if (force and clip.id in picked) or not uses_cache(model) else cache.get(model, clip.id, clip.digest)
            if hit:
                cell.update(status="done", text=hit["text"], seconds=round(hit["seconds"], 2), cached=True,
                            rtf=round(hit["seconds"] / clip.seconds, 2), **_scored(clip, hit["text"]))
            elif (cached_only and uses_cache(model)) or clip.id not in picked:
                cell.update(status="skipped", note="not transcribed yet" + (" (--cached-only)" if cached_only else ""))
            elif not clip.aggregate and not include_full and uses_cache(model):
                cell.update(status="skipped", note="whole-story clip: run with --with-full to transcribe it")
            else:
                todo[model].append(clip)
    todo = {m: t for m, t in todo.items() if t}

    # Progress for the whole queue, per model: what's left and how long one transcription takes (this run's timings once
    # there are some, else the cached ones, never less than a paced API's gap), so the page can show one bar and a time
    # estimate. Local models run one after another; hosted ones run alongside them.
    def pace_of(model: str, timings: list[float]) -> float | None:
        if not timings:
            timings = [c["seconds"] for c in doc["cells"][model].values() if c.get("cached") and c.get("seconds")]
        mean = sum(timings) / len(timings) if timings else None
        return max(mean or 0.0, call_gap(model)) or None

    took_here: dict[str, list[float]] = {m: [] for m in todo}
    doc["work"] = {"total": sum(map(len, todo.values())), "done": 0,
                   "models": {m: {"total": len(t), "done": 0, "local": is_local(m), "state": "queued",
                                  "secPerClip": pace_of(m, [])} for m, t in todo.items()}}
    doc["current"] = []
    flush()

    def finish_cell(model: str, took: float | None = None) -> None:
        w = doc["work"]["models"][model]
        with lock:
            w["done"] += 1
            doc["work"]["done"] += 1
            if took is not None:
                took_here[model].append(took)
                w["secPerClip"] = pace_of(model, took_here[model])
        flush(force=False)

    def work_batch(model: str, batch) -> None:
        """A remote GPU model: every clip it still needs goes out in one call."""
        clips_left = todo[model]
        with lock:
            doc["current"] = doc["current"] + [{"model": model, "clip": f"{len(clips_left)} clips in one batch"}]
            for c in clips_left:
                doc["cells"][model][c.id]["status"] = "running"
        flush()
        print(f"[{model}] sending {len(clips_left)} clips in one batch", flush=True)
        try:
            t0 = time.time()
            out = batch([str(c.wav) for c in clips_left])
            per = (time.time() - t0) / len(clips_left)  # wall time incl. GPU start + model load, spread over the batch
        except Exception as e:
            traceback.print_exc()
            for c in clips_left:
                with lock:
                    doc["cells"][model][c.id].update(status="error", error=f"batch failed: {type(e).__name__}: {e}")
                finish_cell(model)
            return
        for c in clips_left:
            cache.put(model, c.id, c.digest, {"text": out[str(c.wav)], "seconds": per, "batched": len(clips_left)})
            with lock:
                doc["cells"][model][c.id].update(status="done", text=out[str(c.wav)], seconds=round(per, 2), cached=False,
                                                 rtf=round(per / c.seconds, 2), **_scored(c, out[str(c.wav)]))
            finish_cell(model, per)

    def work(model: str) -> None:
        """Transcribe one model's queue. Local models run in turn on the main thread; each hosted one in its own thread."""
        w = doc["work"]["models"][model]
        with lock:
            w["state"] = "running"
        batch = get_batcher(model)
        transcribe, pace, halted = None, get_pacer(model), None
        if batch:
            work_batch(model, batch)
        for clip in [] if batch else todo[model]:
            cell = doc["cells"][model][clip.id]
            if halted:
                with lock:
                    cell.update(status="skipped", note=halted)
                finish_cell(model)
                continue
            with lock:
                cell["status"] = "running"
                doc["current"] = [c for c in doc["current"] if c["model"] != model] + [{"model": model, "clip": clip.id}]
            flush(force=False)
            took = None
            try:
                if transcribe is None:
                    transcribe = get_transcriber(model)
                if pace:
                    pace()  # waits out the API's gap; deliberately outside the timer
                t0 = time.time()
                text = transcribe(str(clip.wav))
                took = time.time() - t0
                if uses_cache(model):
                    cache.put(model, clip.id, clip.digest, {"text": text, "seconds": took})
                with lock:
                    cell.update(status="done", text=text, seconds=round(took, 2), cached=False,
                                rtf=round(took / clip.seconds, 2), **_scored(clip, text))
            except Exception as e:  # one bad cell must not end the run
                with lock:
                    if getattr(e, "skip_cell", False):
                        cell.update(status="skipped", note=str(e))
                    elif getattr(e, "halt_model", False):
                        halted = str(e)
                        cell.update(status="skipped", note=halted)
                        print(f"[{model}] stopping for this run: {e}", flush=True)
                    else:
                        cell.update(status="error", error=f"{type(e).__name__}: {e}")
                        traceback.print_exc()
            finish_cell(model, took)
        with lock:
            w["state"] = "stopped" if halted else "done"
            doc["current"] = [c for c in doc["current"] if c["model"] != model]
        flush()

    hosted = [threading.Thread(target=work, args=(m,), daemon=True) for m in todo if not is_local(m)]
    for t in hosted:
        t.start()
    for model in (m for m in todo if is_local(m)):  # one local model in memory at a time
        work(model)
        unload_local()
    for t in hosted:
        t.join()
    doc.update(state="finished", current=[], finishedAt=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    flush()
    _write(RUNS_DIR / f"{run_id}.json", doc)
    with (RUNS_DIR / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "set", "clip", "status", "cer", "cer_equiv", "ref_len", "seconds", "hypothesis"])
        sets = {c["id"]: c["set"] for c in doc["clips"]}
        for m, cells in doc["cells"].items():
            for cid, c in cells.items():
                w.writerow([m, sets[cid], cid, c["status"], c.get("cer"), c.get("cerEquiv"), c.get("refLen"),
                            c.get("seconds"), c.get("text", c.get("error", ""))])
    return doc
