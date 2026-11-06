"""Fills clips/ with the One reader set (Shanghai Spoken Corpus sentences read aloud) from the published dataset, as
ssc-<file>-p<line>.wav + .txt. Like the Cantonese evals' scripts/fetch_lida.py.

  hf auth login                       # the dataset is gated: request access on its page first (evaluation use only)
  uv run python scripts/fetch_ssc.py

The audio is one person's voice and its licence allows evaluation only: no training, voice cloning or redistribution.
"""
import csv
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "jdhe1120/shanghainese-ssc-read-speech"
OUT = Path(__file__).resolve().parent.parent / "clips"


def main() -> None:
    try:
        root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns="test/*")) / "test"
    except Exception as e:  # noqa: BLE001 - gated: the usual failure is no login or no access yet
        raise SystemExit(f"couldn't download {REPO} ({type(e).__name__}). It is gated: request access at "
                         f"https://huggingface.co/datasets/{REPO}, then `hf auth login`.")
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader((root / "metadata.csv").open(encoding="utf-8")))
    for r in rows:
        shutil.copyfile(root / r["file_name"], OUT / f"{r['id']}.wav")
        (OUT / f"{r['id']}.txt").write_text(r["text"] + "\n", encoding="utf-8")
    print(f"{len(rows)} clips in {OUT}")


if __name__ == "__main__":
    main()
