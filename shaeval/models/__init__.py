"""Model registry. Each module exposes `transcribe(wav_path) -> str`.
Imported lazily so listing models never loads torch."""
import importlib
from typing import Callable

# name -> module under shaeval.models. Only models that claim Wu / Shanghainese.
REGISTRY = {
    "qwen3-asr-stock": "qwen_stock",
    "cn-multidialect": "cn_multidialect",
    "fun-asr-nano": "fun_asr_nano",
    "dolphin-cn": "dolphin_cn",
    "whisper-wu": "whisper_wu",
    "u2pp-wu": "u2pp_wu",
    "step-audio2-wu": "step_audio2_wu",  # Modal GPU, costs money: run explicitly
    "qwen3-omni": "qwen3_omni",  # Modal GPU (A100 80 GB), Qwen's Chinese ASR prompt
    "qwen3-omni-wu": "qwen3_omni_wu",  # same weights, prompted for Shanghainese
    "azure-wuu": "azure_wuu",
    "gemini-argon": "gemini_argon",  # Gemini 4 Argon prompted to transcribe; paid
}
DEFAULT = ["whisper-wu", "qwen3-asr-stock"]
PAID = {"step-audio2-wu", "qwen3-omni", "qwen3-omni-wu", "azure-wuu", "gemini-argon"}  # cost money per run: `--models all` leaves them out; name them explicitly
CLOSED = {"azure-wuu", "gemini-argon"}
# Display order on the page, in the README tables and the chart (as in the Canto evals): open models first, then closed
# APIs; within those, by average rank across the three sets (2026-11-05: Bench, phone, SSC), except that models on the
# same base model sit together (INFO "family": base model, then the Wu training data). Fixed on purpose, so
# columns don't jump around between runs; update by hand if the picture changes.
COLUMNS = ["step-audio2-wu", "u2pp-wu", "whisper-wu",
           "cn-multidialect", "qwen3-asr-stock",       # Qwen3-ASR-1.7B and its dialect fine-tune: kept together
           "dolphin-cn", "fun-asr-nano",
           "qwen3-omni-wu", "qwen3-omni",              # Qwen3-ASR's 30B parent, two prompts; last of the open models (2026-11-05)
           "azure-wuu", "gemini-argon"]


def in_column_order(models) -> list[str]:
    return sorted(models, key=lambda m: COLUMNS.index(m) if m in COLUMNS else len(COLUMNS))


def get_transcriber(name: str) -> Callable[[str], str]:
    return importlib.import_module(f"shaeval.models.{REGISTRY[name]}").transcribe


def uses_cache(name: str) -> bool:
    """False for models that just read a local file (re-reading it is free and picks up edits)."""
    return getattr(importlib.import_module(f"shaeval.models.{REGISTRY[name]}"), "USE_CACHE", True)


def get_batcher(name: str) -> Callable[[list[str]], dict[str, str]] | None:
    """Optional `transcribe_batch(paths) -> {path: text}` for remote GPU models: one call for all uncached clips."""
    return getattr(importlib.import_module(f"shaeval.models.{REGISTRY[name]}"), "transcribe_batch", None)


def get_pacer(name: str) -> Callable[[], None] | None:
    """Optional `pace()` hook: blocks until the model's API will accept another call."""
    return getattr(importlib.import_module(f"shaeval.models.{REGISTRY[name]}"), "pace", None)


def is_local(name: str) -> bool:
    """Runs on this machine, so one at a time; the paid ones are hosted (Azure, the Modal GPU) and run alongside."""
    return name not in PAID


def call_gap(name: str) -> float:
    """Seconds a paced API waits between calls (optional `gap_s()` hook), for the time estimate."""
    f = getattr(importlib.import_module(f"shaeval.models.{REGISTRY[name]}"), "gap_s", None)
    return f() if f else 0.0


def unload_local() -> None:
    """Drop every loaded local model, so the next one gets the memory: in-process weights, then isolated workers."""
    import gc
    import sys
    for mod, attr in (("_qwen", "_loaded"), ("fun_asr_nano", "_loaded"), ("dolphin_cn", "_model")):
        m = sys.modules.get(f"shaeval.models.{mod}")
        if m is not None:
            setattr(m, attr, {} if isinstance(getattr(m, attr), dict) else None)
    if (w := sys.modules.get("shaeval.models._worker")) is not None:
        w.stop_all()
    gc.collect()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Display names. `label` is the short name in tables, `detail` says exactly what ran.
# `hf` is the Hugging Face repo; the module pins its REVISION, which the run records.
INFO = {
    "qwen3-asr-stock": {
        "label": "Qwen3-ASR-1.7B (stock)",
        "family": "Qwen3-ASR-1.7B, stock",
        "detail": "Qwen/Qwen3-ASR-1.7B-hf, untuned. Wu is one of its 22 Chinese dialects; no Wu language name exists, "
                  "so language is auto-detected.",
        "hf": "Qwen/Qwen3-ASR-1.7B-hf",
    },
    "cn-multidialect": {
        "label": "CN-MultiDialect-ASR (Qwen3-ASR-1.7B, dialect-tuned)",
        "family": "Qwen3-ASR-1.7B, tuned on dialect data",
        "detail": "ASLP-lab/CN-MultiDialect-ASR: Qwen3-ASR-1.7B + continual pre-training, dialect SFT and on-policy "
                  "self-distillation (ASLP@NPU, WenetSpeech-Wu authors). qwen-asr 0.0.6 in an isolated env, auto language.",
        "hf": "ASLP-lab/CN-MultiDialect-ASR",
    },
    "fun-asr-nano": {
        "label": "Fun-ASR-Nano-2512",
        "family": "Fun-ASR (Tongyi)",
        "detail": "FunAudioLLM/Fun-ASR-Nano-2512-hf (Tongyi, Dec 2025), transformers-native, language zh "
                  "(7 dialect groups incl. Wu; no Wu setting), greedy, float32.",
        "hf": "FunAudioLLM/Fun-ASR-Nano-2512-hf",
    },
    "dolphin-cn": {
        "label": "Dolphin-CN-Dialect small",
        "family": "Dolphin (DataoceanAI)",
        "detail": "DataoceanAI Dolphin-CN-Dialect small.cn (0.4 B, May 2026) via dataoceanai-dolphin, "
                  "lang_sym zh, region_sym SHANGHAI.",
        "hf": None,  # loaded from ModelScope by the dolphin package
    },
    "whisper-wu": {
        "label": "Whisper-medium-Wu (Whisper-medium, Wu-tuned)",
        "family": "Whisper-medium, tuned on WenetSpeech-Wu",
        "detail": "ASLP-lab/WenetSpeech-Wu-Speech-Understanding whisper/whisper-medium.pt (sha256 e2755246), Whisper medium "
                  "fine-tuned on WenetSpeech-Wu; WeNet a54b90b, default decode, CPU.",
        "hf": None,  # not in the HF cache: scripts/fetch_models.py
    },
    "u2pp-wu": {
        "label": "Conformer-U2pp-Wu (Conformer U2++, Wu only)",
        "family": "Conformer, trained only on WenetSpeech-Wu",
        "detail": "ASLP-lab/WenetSpeech-Wu-Speech-Understanding u2++/u2++.pt (sha256 9afd6fd8), Conformer U2++ trained "
                  "from scratch on WenetSpeech-Wu; WeNet a54b90b, default decode, CPU, dither 0.",
        "hf": None,
    },
    "step-audio2-wu": {
        "label": "Step-Audio2-Wu-ASR (Step-Audio-2-mini, Wu-tuned)",
        "family": "Step-Audio-2-mini, tuned on WenetSpeech-Wu",
        "detail": "stepfun-ai/Step-Audio-2-mini (8.3B) + ASLP-lab WenetSpeech-Wu stepaudio2-ASR LoRA, ms-swift 3.9.3 PtEngine, "
                  "prompt <audio>语音说了什么, greedy, bf16, on Modal A100 40 GB (batched).",
        "hf": None,
    },
    "qwen3-omni": {
        "label": "Qwen3-Omni-30B-A3B (stock, Qwen's Chinese prompt)",
        "family": "Qwen3-Omni-30B-A3B, stock",
        "detail": "Qwen/Qwen3-Omni-30B-A3B-Instruct (30B MoE, 3B active; the model Qwen3-ASR was trained from), audio "
                  "output off, transformers 5.19, prompt 请将这段中文语音转换为纯文本。 (the model card's Chinese ASR "
                  "prompt; no language setting exists), greedy, bf16, on Modal A100 80 GB (batched).",
        "hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    },
    "qwen3-omni-wu": {
        "label": "Qwen3-Omni-30B-A3B + Shanghainese prompt",
        "family": "Qwen3-Omni-30B-A3B, stock",
        "detail": "Same weights and setup as the Chinese-prompt row, prompt 请将这段上海话录音逐字转写成上海话口语。 "
                  "(the Cantonese evals' GLM-ASR Cantonese prompt with the language swapped).",
        "hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    },
    "azure-wuu": {
        "label": "Azure Speech (wuu-CN)",
        "family": "Microsoft, model not named",
        "detail": "Azure Speech fast transcription, standard recognizer, locale wuu-CN (Chinese, Wu, Simplified), "
                  "api-version 2025-10-15. Microsoft doesn't name the model.",
        "hf": None,
    },
    "gemini-argon": {
        "label": "Gemini 4 Argon (prompted)",
        "family": "Google general model, prompted",
        "detail": "gemini-4-argon via generateContent, audio inline with a prompt asking for verbatim spoken Shanghainese "
                  "in simplified characters (gemini_argon.PROMPT, the Cantonese evals' prompt with the language swapped), "
                  "default sampling. Google names no speech languages for it.",
        "hf": None,
    },
}


def describe(name: str) -> dict:
    """INFO entry plus the pinned Hugging Face revision (the module's REVISION), if it has one."""
    info = dict(INFO[name], closed=name in CLOSED)
    rev = getattr(importlib.import_module(f"shaeval.models.{REGISTRY[name]}"), "REVISION", None)
    if rev:
        info["revision"] = rev[:7]
    return info
