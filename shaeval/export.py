"""Writes a small, committable snapshot of the latest run to results/ (runs/ stays out of git):
  results/summary.md   mean per-clip CER per set for each model, in all four views
  results/ranking.svg  the README chart: the headline view, one panel per set
  results/summary.csv  the same numbers plus combined CER, one row per set x model x view
  results/per_clip.csv every model's scores on every clip; reference and transcript only for sets in TEXT_SETS

Only sets in TEXT_SETS may publish text: MagicData's licence forbids reproducing any part of the data. SSC (our own
recordings of U Alberta corpus sentences) publishes its text. A set stays out of summary.md until it has MIN_CLIPS clips. Sets are never pooled, so there is no overall average.
"""
import csv
import json
from pathlib import Path

from . import metrics
from .models import CLOSED, in_column_order
from .runner import RUNS_DIR, flags

OUT = Path(__file__).resolve().parent.parent / "results"
README = OUT.parent / "README.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"  # the README block between these is regenerated
NAMES = {"wubench": "Podcasts & storytelling", "magicdata": "Phone conversation", "ssc": "One reader"}  # table order
TEXT_SETS = {"wubench", "ssc"}
MIN_CLIPS = 100
VIEWS = [("equivNum", "Spellings folded, digits → 中文"), ("equiv", "Spellings folded"),
         ("num", "Strict, digits → 中文"), ("strict", "Strict (one spelling convention)")]


def _chart(names: list[str], panels: list[tuple[str, list[float | None]]], is_open: list[bool], partial: list[bool],
           subtitle: str) -> str:
    """Small multiples on its own light surface (readable on GitHub's light and dark pages): one panel of horizontal bars
    per set, the same model order and one shared scale, so bars compare across panels. Bars colored open / closed."""
    left, panel_w, gap, top, step, thick = max(len(n) for n in names) * 8 + 32, 250, 70, 128, 30, 18
    width = left + len(panels) * panel_w + (len(panels) - 1) * gap + 80  # room for the last panel's value labels
    height = top + step * len(names) + 40
    vmax = max(v for _, vals in panels for v in vals if v is not None)
    scale = panel_w / vmax
    ink, muted, line = "#0b0b0b", "#52514e", "#e4e3df"
    colour = {True: "#2a78d6", False: "#eb6834"}
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;")
    text = lambda x, y, t, size=13, fill=ink, anchor="start", weight=400: (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(t)}</text>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="{esc(subtitle)}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect width="{width}" height="{height}" rx="8" fill="#fcfcfb"/>',
           text(20, 32, "Character error rate: lower is better", 17, weight=600), text(20, 54, subtitle, 13, muted)]
    for x, op, name in ((20, True, "Open model"), (120, False, "Closed API")):
        out += [f'<rect x="{x}" y="66" width="12" height="12" rx="3" fill="{colour[op]}"/>', text(x + 18, 77, name, 13, muted)]
    for i, name in enumerate(names):
        out.append(text(left - 12, top + i * step + 14, name, 13, anchor="end"))
    for k, (title, vals) in enumerate(panels):
        x0 = left + k * (panel_w + gap)
        out += [text(x0, top - 14, title, 13, weight=600),
                f'<line x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{top + step * len(names) - 8}" stroke="{line}"/>']
        for i, v in enumerate(vals):
            if v is None:
                continue
            y, w = top + i * step, max(v * scale, 6)
            out += [f'<path d="M{x0} {y}h{w - 4:.1f}a4 4 0 0 1 4 4v{thick - 8}a4 4 0 0 1 -4 4h-{w - 4:.1f}z" fill="{colour[is_open[i]]}"/>',
                    text(x0 + w + 6, y + 14, f"{v:.1f}%" + (" ◐" if partial[i] else ""), 12)]
    if any(partial):
        out.append(text(20, height - 14, "◐ run on only part of the set.", 12, muted))
    return "\n".join(out + ["</svg>"]) + "\n"


def export(src: Path = RUNS_DIR / "latest.json", out: Path = OUT, readme: Path = README, hide_models=(),
           text_sets=TEXT_SETS) -> Path:
    """hide_models are left out entirely; only text_sets keep their reference and transcript text."""
    doc = json.loads(src.read_text(encoding="utf-8"))
    models = [m for m in doc["models"] if m not in hide_models]
    cells, info = doc["cells"], doc["modelInfo"]
    label = lambda m: info[m]["label"].split(" (")[0]
    out.mkdir(exist_ok=True)
    clips = [c for c in doc["clips"] if c["aggregate"]]
    by_set = {s: [c for c in clips if c["set"] == s] for s in NAMES}
    shown = [s for s, members in by_set.items() if len(members) >= MIN_CLIPS]

    stats = {}
    for s, members in by_set.items():
        for m in models:
            done = [cells[m][c["id"]] for c in members if cells[m][c["id"]]["status"] == "done"]
            for view, _ in VIEWS:
                stats[s, m, view] = metrics.summarize([d["scores"][view] for d in done]) | {"total": len(members)}

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "model", "view", "mean_cer", "corpus_cer", "clips_done", "clips_in_set"])
        for (s, m, view), x in stats.items():
            r = lambda v: None if v is None else round(v, 4)
            w.writerow([s, m, view, r(x["meanCer"]), r(x["corpusCer"]), x["n"], x["total"]])

    fl = [f for f in flags() if f["model"] in models]
    flag_of = lambda m, s: next((f for f in fl if f["model"] == m and f["set"] in (s, "*")), None)

    def cell(s, m, view):
        x = stats[s, m, view]
        if x["meanCer"] is None:
            return "–"
        if flag_of(m, s):
            return f"{100 * x['meanCer']:.1f}% ‡" + ("" if x["n"] == x["total"] else f" ◐ ({x['n']}/{x['total']})")
        return f"{100 * x['meanCer']:.1f}%" + ("" if x["n"] == x["total"] else f" ◐ ({x['n']}/{x['total']})")

    partial = any(stats[s, m, v]["n"] < stats[s, m, v]["total"] for s in shown for m in models for v, _ in VIEWS
                  if stats[s, m, v]["n"])
    md = [f"# Results\n\nRun `{doc['runId']}`. Mean per-clip CER. Sets are scored separately and never pooled. Rows: open "
          "models first, then closed APIs, each roughly by average rank across the sets (a fixed order, `COLUMNS` in "
          "shaeval/models/__init__.py)." + (" ◐ the model ran on only part of the set (n/N), so its number is on fewer "
          "clips than the others'." if partial else "") + " Generated by `python -m shaeval export`.\n"]
    for view, title in VIEWS:
        md += [f"## {title}\n", "| Model | Built on | Type | " + " | ".join(f"{NAMES[s]} ({len(by_set[s])})" for s in shown) + " |",
               "|---|---|---|" + "---|" * len(shown)]
        for m in in_column_order(models):
            md.append(f"| {label(m)} | {info[m].get('family', '')} | {'closed' if m in CLOSED else 'open'} | "
                      + " | ".join(cell(s, m, view) for s in shown) + " |")
        md.append("")
    view = VIEWS[0][0]
    order = in_column_order(models)
    (out / "ranking.svg").write_text(_chart(
        [label(m) for m in order],
        [(f"{NAMES[s]} ({len(by_set[s])} clips)", [None if (v := stats[s, m, view]["meanCer"]) is None else 100 * v for m in order])
         for s in shown],
        [m not in CLOSED for m in order],
        [any(stats[s, m, view]["n"] < stats[s, m, view]["total"] for s in shown) for m in order],
        "Mean per-clip CER, spellings folded, digits as Chinese numerals. Sets scored separately."), encoding="utf-8")
    # README: best per set, the chart and the headline table, between the markers
    if readme.exists() and BEGIN in (text := readme.read_text(encoding="utf-8")) and END in text:
        best = ["| Test set | Best | Within a point |", "|---|---|---|"]
        for s in shown:
            scored = sorted((v, m) for m in models if (v := stats[s, m, view]["meanCer"]) is not None
                            and stats[s, m, view]["n"] == stats[s, m, view]["total"])
            near = [f"{label(m)} {cell(s, m, view)}" for v, m in scored[1:3] if v - scored[0][0] <= 0.01]
            best.append(f"| {NAMES[s]} | **{label(scored[0][1])}** {cell(s, scored[0][1], view)} | {', '.join(near) or '–'} |")
        table = ["| Model | Built on | Type | " + " | ".join(f"{NAMES[s]} ({len(by_set[s])})" for s in shown) + " |",
                 "|---|---|---|" + "---|" * len(shown)]
        table += [f"| {label(m)} | {info[m].get('family', '')} | {'closed' if m in CLOSED else 'open'} | " + " | ".join(cell(s, m, view) for s in shown)
                  + " |" for m in order]
        new = [BEGIN, *best, "", "## Overall", "", "![Models ranked by character error rate](results/ranking.svg)", "",
               *table, "",
               "Mean per-clip character error rate, spellings folded, digits → 中文. Open models first, then closed APIs, better models first "
               "within each; models on the same base model sit together. ‡ familiar data: the set closely resembles "
               "that model's training data, so its score is probably better than on new audio (see below)." + (" ◐ only part of the set (n/N)." if partial else "") + " Every view, every "
               f"Podcasts & storytelling transcript, and the notes behind each ‡ are in [`results/`](results/summary.md) (run `{doc['runId']}`).", END]
        head, rest = text.split(BEGIN, 1)
        readme.write_text(head + "\n".join(new) + rest.split(END, 1)[1], encoding="utf-8")

    md += ["## Familiar data\n", "‡ marks familiar data: evidence the set closely resembles the model's training data, but "
           "not the clear sign of training on the test clips themselves (rules/flags.toml).\n"]
    md += [f"- **{label(f['model'])}**, {NAMES.get(f['set'], f['set'])}: {f['note']}" for f in fl] + [""]
    md += ["## Models\n"] + [f"- **{label(m)}**: {info[m].get('detail', '')}" for m in models]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (out / "per_clip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "clip", "seconds", "reference", "model", "transcript", "cer_folded", "cer_strict"]
                   + [f"{k}_{v}" for v, _ in VIEWS for k in ("edits", "chars")])
        for c in clips:
            ok = c["set"] in text_sets
            for m in models:
                x = cells[m][c["id"]]
                if x["status"] == "done":
                    sc = x["scores"]
                    w.writerow([c["set"], c["id"], c["seconds"], c["reference"] if ok else "", m, x["text"] if ok else "",
                                round(sc["equiv"]["cer"], 4), round(sc["strict"]["cer"], 4)]
                               + [sc[v][k] for v, _ in VIEWS for k in ("edits", "refLen")])
    return out
