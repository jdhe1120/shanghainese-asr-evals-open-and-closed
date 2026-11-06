"""FunAudioLLM/Fun-ASR-Nano-2512-hf (Tongyi, Dec 2025), transformers-native port (needs transformers >= 5.17).

Chinese covers 7 dialect groups including Wu. `language` only takes zh/en/ja, so it is set to zh (Chinese); there is no
Wu setting. Audio is loaded mono 16 kHz here, as the model card asks for local arrays.
"""
import librosa
import torch

from ._qwen import device

MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
REVISION = "7ef6cafdc445b5beff0b730cbeb51952c793ca6e"  # pinned
_loaded = None


def transcribe(wav_path: str) -> str:
    global _loaded
    if _loaded is None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        # float32: the card only validates CPU float32; MPS half precision is untested for this model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).to(device()).eval()
        _loaded = (model, proc)
    model, proc = _loaded
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = proc.apply_transcription_request(
        audio=audio, language="zh",
        processor_kwargs={"return_tensors": "pt", "audio_kwargs": {"sampling_rate": 16000}},
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max(128, int(len(audio) / 16000 * 8)), do_sample=False)
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
