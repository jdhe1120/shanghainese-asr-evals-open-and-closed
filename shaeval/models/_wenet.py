"""Shared runner for WeNet checkpoints: one isolated worker process per model (_wenet_worker.py), WeNet installed from
the GitHub archive of commit a54b90b (the commit shanghainese-transcription vendors; `git+` installs fail on a broken
submodule entry), plus torchcodec, which WeNet's wav decoding goes through.

A model dir must hold final.pt, train.yaml (with input_dim / output_dim) and units.txt: wenet.cli.model.load_model's
layout. scripts/fetch_models.py puts them in models/ (gitignored); .env can point elsewhere.
"""
from pathlib import Path

from ._env import ROOT, env
from ._worker import Worker

WENET = "wenet @ https://github.com/wenet-e2e/wenet/archive/a54b90bc768679bd4217e4c7765c0671fbfb3a7a.zip"
WORKER = Path(__file__).with_name("_wenet_worker.py")
MODELS = ROOT / "models"


def transcriber(name: str, env_var: str, default_dir: Path):
    worker = None

    def transcribe(wav_path: str) -> str:
        nonlocal worker
        if worker is None:
            model_dir = Path(env().get(env_var) or default_dir)
            missing = [f for f in ("final.pt", "train.yaml", "units.txt") if not (model_dir / f).exists()]
            if missing:
                raise RuntimeError(f"{name} not found in {model_dir} (missing {missing}): run "
                                   f"scripts/fetch_models.py {name}, or set {env_var} in .env")
            worker = Worker(name, ["uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", WENET,
                                   "--with", "torch", "--with", "torchaudio", "--with", "torchcodec",
                                   "python", "-I", str(WORKER), str(model_dir)])
        return worker.transcribe(wav_path)

    return transcribe
