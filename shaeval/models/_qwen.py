"""Shared loader for Qwen3-ASR checkpoints in transformers' -hf format; also device()/dtype() for the other local models.

Qwen3-ASR has no Wu/Shanghainese language name (its forced-language list is 30 languages; Wu is one of the 22
dialects it detects under Chinese), so the language is left to auto-detection and the "language X<asr_text>" prefix
the model writes is stripped by the decode call."""
from pathlib import Path

import librosa
import torch

_loaded: dict[str, tuple] = {}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype():
    d = device()
    if d == "mps":
        return torch.bfloat16
    if d == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def transcribe(model_id: str, wav_path: str, revision: str) -> str:
    if model_id not in _loaded:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        proc = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=dtype()).to(device()).eval()
        _loaded[model_id] = (model, proc)
    model, proc = _loaded[model_id]
    audio, _ = librosa.load(str(Path(wav_path)), sr=16000, mono=True)
    inputs = proc.apply_transcription_request(audio=audio, language=None)
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max(200, int(len(audio) / 16000 * 8)), do_sample=False)
    # transcription_only strips the "language X<asr_text>" prefix and collapses runaway repetition.
    return proc.decode(out[:, inputs.input_ids.shape[1]:], return_format="transcription_only",
                       clean_up_tokenization_spaces=False)[0].strip()
