"""Step-Audio2-Wu-ASR on Modal: stepfun-ai/Step-Audio-2-mini (8.3B, bf16) + the WenetSpeech-Wu ASR LoRA
(ASLP-lab/WenetSpeech-Wu-Speech-Understanding/stepaudio2-ASR), run the way the lab documents (`swift infer`, ms-swift's
PtEngine): user turn "<audio>语音说了什么", greedy (their training args: temperature 0, num_beams 1).

Versions: ms-swift 3.9.3 (the 3.x line the adapter was trained in; its model table requires transformers==4.53.3 for
step_audio2_mini). torch/torchaudio 2.6 so torchaudio.load works without torchcodec.

Cost (2026-10-13 pricing): A100 40 GB at $0.000583/s. Weights are downloaded once into the `shanghainese-weights`
Volume by a CPU-only function. Called from shaeval/models/step_audio2_wu.py; can also be run directly:

    modal run modal_apps/step_audio2_wu.py::download          # one time, CPU only
    modal run modal_apps/step_audio2_wu.py --wav clips/a0002-p12.wav   # one-clip smoke test
"""
import modal

APP_NAME = "shanghainese-step-audio2-wu"
BASE = "/weights/Step-Audio-2-mini"
ADAPTER = "/weights/stepaudio2-ASR"
PROMPT = "<audio>语音说了什么"
GPU = "A100-40GB"
BATCH = 16

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("shanghainese-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "sox", "libsox-dev")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "transformers==4.53.3", "ms-swift==3.9.3", "peft",
                 "accelerate", "librosa", "soundfile", "huggingface_hub")
)


@app.function(image=image, volumes={"/weights": weights}, timeout=3600, cpu=4)
def download() -> list[str]:
    """CPU only: fetch the base model and the Wu ASR adapter into the Volume (skips files already there)."""
    import os

    from huggingface_hub import snapshot_download

    snapshot_download("stepfun-ai/Step-Audio-2-mini", revision="e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36", local_dir=BASE, ignore_patterns=["assets/*", "token2wav/*"])
    snapshot_download("ASLP-lab/WenetSpeech-Wu-Speech-Understanding", revision="cb53da7aefd9acd0f2af61b9631e49f5c3382686",
                      local_dir="/weights/_wswu",
                      allow_patterns=["stepaudio2-ASR/adapter_config.json", "stepaudio2-ASR/adapter_model.safetensors",
                                      "stepaudio2-ASR/additional_config.json", "stepaudio2-ASR/args.json"])
    os.makedirs(ADAPTER, exist_ok=True)
    for f in os.listdir("/weights/_wswu/stepaudio2-ASR"):
        os.replace(f"/weights/_wswu/stepaudio2-ASR/{f}", f"{ADAPTER}/{f}")
    weights.commit()
    return sorted(os.listdir(BASE)) + [f"adapter/{f}" for f in sorted(os.listdir(ADAPTER))]


@app.function(image=image, volumes={"/weights": weights}, gpu=GPU, timeout=1800, cpu=4, memory=32768)
def transcribe(items: list[tuple[str, bytes]]) -> dict[str, str]:
    """items: (clip id, wav bytes). Returns clip id -> text. One model load, clips run BATCH at a time."""
    import os
    import tempfile
    import time

    from swift.llm import InferRequest, PtEngine, RequestConfig

    t0 = time.time()
    engine = PtEngine(BASE, adapters=[ADAPTER], torch_dtype="bfloat16", max_batch_size=BATCH)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    tmp = tempfile.mkdtemp()
    requests, ids = [], []
    for cid, data in items:
        path = os.path.join(tmp, f"{cid}.wav")
        with open(path, "wb") as f:
            f.write(data)
        requests.append(InferRequest(messages=[{"role": "user", "content": PROMPT}], audios=[path]))
        ids.append(cid)
    t1 = time.time()
    responses = engine.infer(requests, RequestConfig(max_tokens=512, temperature=0))
    print(f"{len(ids)} clips in {time.time() - t1:.0f}s", flush=True)
    return {cid: r.choices[0].message.content for cid, r in zip(ids, responses)}


@app.local_entrypoint()
def main(wav: str):
    from pathlib import Path

    p = Path(wav)
    print(transcribe.remote([(p.stem, p.read_bytes())]))
