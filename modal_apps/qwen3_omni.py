"""Qwen3-Omni-30B-A3B-Instruct on Modal, as a transcriber: audio in, text out, the speech-output half (talker) never
loaded. Qwen's model card has no language setting and names no Chinese dialect but Cantonese, so the language is in the
prompt: the card's own Chinese ASR prompt, or a Shanghainese one (shaeval/models/qwen3_omni*.py pass it).

Runs as on the card: user turn = audio, then the prompt text, no system prompt; greedy; transformers (vLLM needs a fork).
30B MoE (3B active), bf16 thinker ~62 GB, so an A100 80 GB.

Called from shaeval/models/qwen3_omni.py and qwen3_omni_wu.py; can also be run directly:

    modal run modal_apps/qwen3_omni.py::download                  # one time, CPU only (~70 GB into the Volume)
    modal run modal_apps/qwen3_omni.py --wav clips/a0002-p12.wav  # one-clip smoke test, Chinese prompt
"""
import modal

APP_NAME = "shanghainese-qwen3-omni"
REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"  # 2025-09-22
PATH = "/weights/Qwen3-Omni-30B-A3B-Instruct"
GPU = "A100-80GB"
BATCH = 8
PROMPT_ZH = "请将这段中文语音转换为纯文本。"  # the model card's Chinese ASR prompt

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("shanghainese-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "pillow", "transformers==5.19.0", "accelerate", "librosa", "soundfile",
                 "huggingface_hub")  # torchvision/pillow: the processor loads its image half even for audio-only use
)


@app.function(image=image, volumes={"/weights": weights}, timeout=7200, cpu=4)
def download() -> list[str]:
    """CPU only: fetch the checkpoint into the Volume (skips files already there)."""
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=REVISION, local_dir=PATH)
    weights.commit()
    return sorted(os.listdir(PATH))


@app.function(image=image, volumes={"/weights": weights}, gpu=GPU, timeout=3600, cpu=4, memory=65536)
def transcribe(items: list[tuple[str, bytes]], prompt: str) -> dict[str, str]:
    """items: (clip id, wav bytes). Returns clip id -> text. One model load, clips run BATCH at a time."""
    import io
    import time

    import librosa
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        PATH, enable_audio_output=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(PATH)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    msg = [{"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    out, t1 = {}, time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        audio = [librosa.load(io.BytesIO(b), sr=16000)[0] for _, b in chunk]
        inputs = proc(text=[text] * len(chunk), audio=audio, return_tensors="pt", padding=True).to(model.device)
        inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=512, do_sample=False, return_audio=False,
                                 thinker_return_dict_in_generate=False)
        gen = gen[0] if isinstance(gen, tuple) else gen
        new = gen[:, inputs["input_ids"].shape[1]:]
        for (cid, _), t in zip(chunk, proc.batch_decode(new, skip_special_tokens=True)):
            out[cid] = t.strip()
        print(f"{min(i + BATCH, len(items))}/{len(items)} clips, {time.time() - t1:.0f}s", flush=True)
    return out


@app.local_entrypoint()
def main(wav: str, prompt: str = PROMPT_ZH):
    from pathlib import Path

    p = Path(wav)
    print(transcribe.remote([(p.stem, p.read_bytes())], prompt))
