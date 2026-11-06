"""Download the models that aren't in the normal Hugging Face cache into models/ (gitignored), at pinned revisions, and
apply the fixes the releases need so they load. Checksums are verified; finished models are skipped.

    uv run python scripts/fetch_models.py [whisper-wu u2pp-wu dolphin-cn]

  whisper-wu   ASLP-lab/WenetSpeech-Wu-Speech-Understanding whisper/ (2.9 GB). WeNet's load_model wants final.pt,
               units.txt and output_dim in train.yaml: whisper-medium.pt -> final.pt, an empty units.txt (the Whisper
               tokenizer doesn't use it), `output_dim: 51865` appended.
  u2pp-wu      the same repo's u2++/ (0.5 GB): u2++.pt -> final.pt, dict/lang_char.txt -> units.txt and
               lang_char.txt (load_model wants the first, the tokenizer the second), `input_dim: 80` and
               `output_dim: 5949` appended (WeNet's train.py normally writes them), fbank dither 1.0 -> 0.0 (training
               noise left on at inference makes the output random).
  dolphin-cn   ModelScope DataoceanAI/dolphin-cn-dialect-small (0.9 GB), unmodified. The dolphin package's own download
               asks for "dolphi-cn-dialect-small" (a typo) and fails; its sha256 check accepts this copy.

The Hugging Face models (Qwen3-ASR, Fun-ASR-Nano, CN-MultiDialect-ASR) download themselves on first use.
"""
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WU_REPO, WU_REV = "ASLP-lab/WenetSpeech-Wu-Speech-Understanding", "cb53da7aefd9acd0f2af61b9631e49f5c3382686"
SHA = {"whisper-wu": "e2755246fd1ccedb9052e953ffdf3ce2a18500bedc97a307be94bafd1dfbb165",
       "u2pp-wu": "9afd6fd82f8c638bdde993764c3c9c04e74fb8db20dc8399fd8fc6b8a34e7670",
       "dolphin-cn": "1cee2b8d2133cabb36567625a832d4033569e27eaf5f98df9be1139ec6068bbb"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, weights: Path) -> None:
    if sha256(weights) != SHA[name]:
        raise SystemExit(f"{name}: {weights} has the wrong sha256; delete it and run again")
    print(f"{name}: ok ({weights.parent})")


def wu_files(subdir: str, files: list[str]) -> Path:
    from huggingface_hub import hf_hub_download
    tmp = MODELS / ".download"
    for f in files:
        hf_hub_download(WU_REPO, f"{subdir}/{f}", revision=WU_REV, local_dir=tmp)
    return tmp / subdir


def whisper() -> None:
    out = MODELS / "whisper-medium-wu" / "whisper"
    if not (out / "final.pt").exists():
        src = wu_files("whisper", ["whisper-medium.pt", "train.yaml"])
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(src / "whisper-medium.pt", out / "final.pt")
        (out / "units.txt").write_text("")
        (out / "train.yaml").write_text((src / "train.yaml").read_text().rstrip("\n") + "\noutput_dim: 51865\n")
    check("whisper-wu", out / "final.pt")


def u2pp() -> None:
    out = MODELS / "conformer-u2pp-wu"
    if not (out / "final.pt").exists():
        src = wu_files("u2++", ["u2++.pt", "train.yaml", "dict/lang_char.txt"])
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(src / "u2++.pt", out / "final.pt")
        shutil.copy(src / "dict" / "lang_char.txt", out / "units.txt")
        shutil.copy(src / "dict" / "lang_char.txt", out / "lang_char.txt")  # the tokenizer looks for its own name too
        yaml, n = re.subn(r"(\n\s+dither:) 1\.0", r"\1 0.0  # training-time noise off for inference",
                          (src / "train.yaml").read_text())
        assert n == 1, "u2++ train.yaml: expected one `dither: 1.0`"
        (out / "train.yaml").write_text(yaml.rstrip("\n") + "\n\n# WeNet's train.py normally writes these; the release "
                                        "omitted them\ninput_dim: 80\noutput_dim: 5949\n")
    check("u2pp-wu", out / "final.pt")


def dolphin() -> None:
    out = MODELS / "dolphin-cn-dialect-small"
    if not (out / "small.cn.pt").exists():
        from modelscope import snapshot_download
        snapshot_download("DataoceanAI/dolphin-cn-dialect-small", local_dir=str(out))
    check("dolphin-cn", out / "small.cn.pt")


if __name__ == "__main__":
    jobs = {"whisper-wu": whisper, "u2pp-wu": u2pp, "dolphin-cn": dolphin}
    for name in sys.argv[1:] or jobs:
        jobs[name]()
    shutil.rmtree(MODELS / ".download", ignore_errors=True)
