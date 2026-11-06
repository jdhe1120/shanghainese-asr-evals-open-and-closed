import argparse
import json
import webbrowser

from . import cache, runner, server
from .clips import SETS, load_clips
from .models import DEFAULT, PAID, REGISTRY, uses_cache


def main() -> None:
    ap = argparse.ArgumentParser(prog="shaeval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="transcribe + score")
    r.add_argument("--models", nargs="+", default=DEFAULT, choices=list(REGISTRY) + ["all"],
                   help="model names, or `all` (all free models; paid ones must be named)")
    r.add_argument("--clips", nargs="+", help="clip ids (default: all)")
    r.add_argument("--set", nargs="+", choices=SETS, dest="sets",
                   help="only these clip sets (magicdata: the corpus; ssc: one reader, Shanghai Spoken Corpus sentences; wubench: WenetSpeech-Wu-Bench). Default: all")
    r.add_argument("--model-clips", nargs="+", default=[], metavar="MODEL=CLIP[,CLIP]",
                   help="limit a model to some clips, e.g. azure-wuu=a0002-p17")
    r.add_argument("--with-full", action="store_true",
                   help="transcribe the whole-story *-full clips too (off by default: the app splits long audio itself); "
                        "without it they still show cached results, outside the averages")
    r.add_argument("--force", action="store_true", help="ignore the hypothesis cache")
    r.add_argument("--cached-only", action="store_true",
                   help="never call a model: rebuild runs/latest.json from cached results (use after parallel runs)")
    r.add_argument("--no-ui", action="store_true")
    r.add_argument("--port", type=int, default=8766)
    u = sub.add_parser("ui", help="serve the UI for the latest run")
    u.add_argument("--port", type=int, default=8766)
    sub.add_parser("export", help="write the committable snapshot of the latest run to results/")
    sub.add_parser("check", help="verify clips, normalisation and deps without loading a model")
    a = ap.parse_args()

    if a.cmd == "check":
        from .metrics import score
        from .normalize import canonical
        clips = load_clips()  # a fresh clone has none yet: the scoring checks below still run
        for c in clips:
            s = score(c.reference, canonical(c.reference))
            assert s["cer"] == 0 and s["refLen"] > 0, c.id
            print(f"{c.id:14s} {c.seconds:6.1f}s {s['refLen']:4d} chars")
        for c in clips:
            assert "[" not in c.reference and "+" not in c.reference, f"{c.id}: annotation tag left in the reference"
        assert score("搿呃辰光", "搿個辰光")["cer"] == 0  # strict holds the canonical spelling: 呃 -> 个, 個 -> 个 by t2s
        assert score("吾也伐晓得", "吾也伐晓得")["cer"] > 0  # ... so the corpus's own spelling is not it (我 勿)
        assert canonical("对伐，吾伐晓得，是伐是？对伐起，晓得伐") == "对伐，我勿晓得，是勿是？对勿起，晓得伐"  # veq vs va
        assert canonical("呃，伊是老好呃") == "呃，伊是老好个"  # clause-initial filler 呃 stays
        assert score("吾伐晓得", "吾伐晓得", equiv=True)["cer"] == 0  # folded still accepts the corpus spelling
        assert score("葛末吾伐晓得呃", "格么我不晓得个", equiv=True)["cer"] == 0  # 葛/格, 末/么, 吾/我, 伐/不, 呃/个
        assert score("末", "么", equiv=True)["cer"] == 0 and score("么", "嘛", equiv=True)["cer"] == 0
        assert score("末", "嘛", equiv=True)["cer"] == 1  # pairs don't chain
        assert score("老好呃", "老好的", equiv=True)["cer"] > 0  # Mandarin 的 for gheh stays an error
        assert score("八七年", "87年", numerals=True)["cer"] == 0
        if not clips:
            print("ok: canonical spelling, t2s, Wu fold pairs (no chaining) and numerals work. No clips yet: see "
                  "'Getting the clips' in the README, then run check again to verify them.")
            return
        n = {s: sum(c.set == s for c in clips) for s in SETS}
        print(f"ok: {len(clips)} clips load ({', '.join(f'{k} {v}' for k, v in n.items())}), canonical self-CER is 0, no tags left, t2s, Wu fold pairs (no chaining) and numerals work")
    elif a.cmd == "export":
        from .export import export
        print(export())
    elif a.cmd == "ui":
        srv = server.serve(a.port)
        print(f"http://127.0.0.1:{a.port}")
        webbrowser.open(f"http://127.0.0.1:{a.port}")
        srv.serve_forever()
    else:
        if not a.no_ui:
            server.serve(a.port, background=True)
            webbrowser.open(f"http://127.0.0.1:{a.port}")
            print(f"UI: http://127.0.0.1:{a.port}")
        mc = {k: v.split(",") for k, v in (x.split("=", 1) for x in a.model_clips)}
        if "all" in a.models:  # paid models only when named (their cached results still show in the UI)
            # --cached-only: every model that has results (a never-run model would only add an empty column)
            named = [m for m in a.models if m != "all"]  # `all azure-wuu` = the free models plus the named paid ones
            a.models = ([m for m in REGISTRY if uses_cache(m) and any((cache.CACHE_DIR / m).glob("*.json"))]
                        if a.cached_only else [m for m in REGISTRY if m not in PAID or m in named])
        doc = runner.run(a.models, a.clips, a.force, mc, a.with_full, a.cached_only, a.sets)
        print(json.dumps(doc["summary"], ensure_ascii=False, indent=1))
        if not a.no_ui:
            print("run finished; UI stays up, Ctrl-C to exit")
            try:
                import time
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass


main()
