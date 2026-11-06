"""Clip set: clips/<id>.wav + clips/<id>.txt pairs.

Three sets, scored separately and never pooled: "magicdata", ids <conversation>-p<line> (scripts/prepare_magicdata.py);
"ssc", ids ssc-<file>-p<line>: the author reading Shanghai Spoken Corpus sentences aloud; and "wubench", ids
wub-p<row>: WenetSpeech-Wu-Bench's ASR test set (scripts/prepare_wubench.py). A clip whose id ends in "-full" (none yet: a whole
conversation) would be the same audio as the others joined up; it is reported but left out of the aggregate.
"""
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

CLIPS_DIR = Path(__file__).resolve().parent.parent / "clips"


@dataclass
class Clip:
    id: str
    wav: Path
    reference: str
    seconds: float
    digest: str  # audio + reference, so a changed clip never reuses a stale cache entry

    @property
    def set(self) -> str:
        return "ssc" if self.id.startswith("ssc-") else "wubench" if self.id.startswith("wub-") else "magicdata"

    @property
    def aggregate(self) -> bool:
        return not self.id.endswith("-full")


SETS = ("magicdata", "ssc", "wubench")


def load_clips(directory: Path = CLIPS_DIR) -> list[Clip]:
    clips = []
    for wav in sorted(directory.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        ref = unicodedata.normalize("NFKC", txt.read_text(encoding="utf-8").strip())
        h = hashlib.sha256(wav.read_bytes() + ref.encode()).hexdigest()[:12]
        clips.append(Clip(wav.stem, wav, ref, sf.info(str(wav)).duration, h))
    # conversation by conversation, segments in order; any full-length clip after them
    def order(c: Clip):
        story, _, tail = c.id.rpartition("-")
        return (not c.aggregate, story, int(tail[1:]) if tail[1:].isdigit() else 0)
    return sorted(clips, key=order)


def clip_set_version(clips: list[Clip]) -> str:
    return hashlib.sha256("".join(c.digest for c in clips).encode()).hexdigest()[:8]
