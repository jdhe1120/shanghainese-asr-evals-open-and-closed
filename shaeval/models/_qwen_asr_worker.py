"""Stand-alone worker for checkpoints in the original Qwen3-ASR format (not -hf), run in its own isolated env because
the `qwen-asr` package pins transformers==4.57.6 and the main env needs transformers >= 5.17 (Fun-ASR-Nano).

Started by cn_multidialect.py; protocol in _worker.py.
"""
import os
import sys

# Protocol replies go to a private copy of stdout; fd 1 itself is pointed at stderr *before* any import, so nothing a
# library prints (WeNet prints at import time, subprocesses inherit fd 1) can land in the reply stream.
out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)

import json  # noqa: E402

import torch
from qwen_asr import Qwen3ASRModel

dev = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = Qwen3ASRModel.from_pretrained(sys.argv[1], dtype=torch.bfloat16 if dev != "cpu" else torch.float32,
                                      device_map=dev, max_inference_batch_size=1, max_new_tokens=512)
print(json.dumps({"ready": dev}), file=out, flush=True)
for line in sys.stdin:
    try:
        r = model.transcribe(audio=line.strip(), language=None)[0]  # auto: the model card's "or None" option
        print(json.dumps({"text": r.text, "language": r.language}, ensure_ascii=False), file=out, flush=True)
    except Exception as e:  # report and keep serving
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=out, flush=True)
