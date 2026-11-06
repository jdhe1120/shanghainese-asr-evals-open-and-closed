"""Rebuilds a run document for the UI from the committed results/per_clip.csv, for a fresh clone with no runs/
(copied from the Cantonese evals).

The CSV keeps each cell's transcript and edit counts; the diffs (most of runs/latest.json's size) are recomputed here.
Clips whose text isn't published (MagicData, SSC: no reference in the CSV) keep their scores and get no diff.
"""
import csv
from collections import defaultdict
from pathlib import Path

from . import metrics
from .export import VIEWS
from .models import describe, in_column_order
from .runner import VARIANTS, _summary, flags

CSV = Path(__file__).resolve().parent.parent / "results" / "per_clip.csv"


def from_results(path: Path = CSV) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    clips, cells = {}, defaultdict(dict)
    for r in rows:
        cid, ref, text = r["clip"], r["reference"], r["transcript"]
        clips.setdefault(cid, {"id": cid, "set": r["set"], "seconds": float(r["seconds"]), "reference": ref,
                               "aggregate": True})
        scores = {}
        for view, _ in VIEWS:
            equiv, numerals = VARIANTS[view]
            edits, chars = int(r[f"edits_{view}"]), int(r[f"chars_{view}"])
            scores[view] = {"edits": edits, "refLen": chars, "cer": edits / chars if chars else 0.0,
                            "diff": metrics.diff(ref, text, equiv, numerals) if ref else None}
        cells[r["model"]][cid] = {"status": "done", "text": text, "cached": True, "scores": scores}
    models = in_column_order(cells)
    doc = {"runId": "results/per_clip.csv", "state": "finished", "current": [], "clipSetVersion": "",
           "models": models, "modelInfo": {m: describe(m) for m in models}, "clips": list(clips.values()),
           "snapshot": True, "flags": flags(),
           "cells": {m: {cid: cells[m].get(cid, {"status": "skipped", "note": "no result in results/per_clip.csv"})
                         for cid in clips} for m in models}}
    doc["summary"] = _summary(doc)
    return doc
