"""Sample clips from WenetSpeech-Wu-Bench's ASR test set (ASLP-lab, Apache-2.0, 4,851 16 kHz utterances, 9.75 h of
podcasts, broadcast and storytelling; Shanghainese and Suzhounese are mixed, with no dialect tag) into
clips/wub-p<row>.wav|txt.

Same filters as prepare_magicdata.py (1.5-20 s, >=4 characters), plus repeated sentences dropped. Sampling is seeded and
the dataset revision is pinned, so re-running gives the same clips; pN is the row in the parquet file, so an id never
changes when --n does.

    uv run python scripts/prepare_wubench.py [--n 300]
"""
import argparse
import io
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "clips"
REPO, REVISION = "ASLP-lab/WenetSpeech-Wu-Bench", "36f67e0276ce1bc2892ade713bdbb26f4fa58c75"
MIN_S, MAX_S, MIN_CHARS = 1.5, 20.0, 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = hf_hub_download(REPO, "understanding/asr.parquet", repo_type="dataset", revision=REVISION,
                           local_dir=ROOT / "datasets" / "wubench")
    usable, seen, rows = [], set(), 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200, columns=["audio", "label"]):
        for r in batch.to_pylist():
            info = sf.info(io.BytesIO(r["audio"]))
            ref = r["label"].strip()
            if (MIN_S <= info.duration <= MAX_S and len(re.sub(r"[\W_]", "", ref)) >= MIN_CHARS
                    and ref not in seen):
                seen.add(ref)
                usable.append(rows)
            rows += 1
    picked = set(random.Random(f"{a.seed}-wubench").sample(usable, a.n))
    row = 0
    OUT.mkdir(exist_ok=True)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200, columns=["audio", "label"]):
        for r in batch.to_pylist():
            if row in picked:
                stem = OUT / f"wub-p{row}"
                stem.with_suffix(".wav").write_bytes(r["audio"])
                stem.with_suffix(".txt").write_text(r["label"].strip() + "\n", encoding="utf-8")
            row += 1
    print(f"{len(picked)} of {len(usable)} usable utterances ({rows} in all) in {OUT}")


if __name__ == "__main__":
    main()
