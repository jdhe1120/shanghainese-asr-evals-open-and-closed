"""Cut utterance clips out of the MagicData Shanghai Dialect Conversational Speech Corpus (sample: 10 two-person
phone conversations, 16 kHz mono, about 4.4 h) into clips/<conversation>-p<line>.wav|txt.

Each TXT line is `[start,end]<TAB>speaker<TAB>gender<TAB>transcript`. A segment is kept only if it is one speaker's
clean speech:
  * no overlap (+ / [+]), no unintelligible [*], no redacted [PII], no lone [FILLER];
  * not the scripted "北京爱数智慧语音采集" line every recording opens with;
  * MIN_S..MAX_S long, with at least MIN_CHARS characters left once noise tags ([LAUGHTER] [ENS] [SONANT] ...) are
    removed from the reference.
Then --per-conv segments per conversation are sampled with a fixed seed, so re-running gives the same clips. The
id's pN is the segment's line number in its TXT file, so an id never changes when the sample size does.

The corpus licence forbids redistribution: corpus and clips stay out of git.

    uv run python scripts/prepare_magicdata.py [--per-conv 30]
"""
import argparse
import random
import re
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "Shanghai_Dialect_Conversational_Speech_Corpus"
OUT = ROOT / "clips"
MIN_S, MAX_S, MIN_CHARS = 1.5, 20.0, 4
LINE = re.compile(r"\[([\d.]+),([\d.]+)\]\t(\S+)\t(\S+)\t(.*)")
DROP = re.compile(r"\+|\[\*\]|\[PII\]|\[FILLER\]|爱数智慧")
NOISE = re.compile(r"\[[A-Z]+\]")


def segments(txt: Path):
    for n, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        m = LINE.match(line)
        if not m:
            continue
        start, end, speaker, text = float(m[1]), float(m[2]), m[3], m[5]
        if DROP.search(text) or speaker == "G0000":
            continue
        ref = NOISE.sub("", text).strip()
        if MIN_S <= end - start <= MAX_S and len(re.sub(r"[\W_]", "", ref)) >= MIN_CHARS:
            yield n, start, end, ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-conv", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    total = 0
    for txt in sorted((CORPUS / "TXT").glob("*.txt")):
        conv = txt.stem.split("_")[0].lower()  # A0002_S0003_0_G0003_G0004 -> a0002
        segs = list(segments(txt))
        picked = sorted(random.Random(f"{a.seed}-{conv}").sample(segs, min(a.per_conv, len(segs))))
        audio, sr = sf.read(CORPUS / "WAV" / f"{txt.stem}.wav", dtype="int16")
        for n, start, end, ref in picked:
            stem = OUT / f"{conv}-p{n}"
            sf.write(stem.with_suffix(".wav"), audio[int(start * sr):int(end * sr)], sr, subtype="PCM_16")
            stem.with_suffix(".txt").write_text(ref + "\n", encoding="utf-8")
        total += len(picked)
        print(f"{conv}: {len(picked)} of {len(segs)} usable segments")
    print(f"{total} clips in {OUT}")


if __name__ == "__main__":
    main()
