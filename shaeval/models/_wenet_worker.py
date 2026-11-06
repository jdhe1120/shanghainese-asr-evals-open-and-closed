"""Stand-alone worker for WeNet checkpoints (Whisper-medium-Wu), run in its own isolated env: WeNet isn't on PyPI and
pins old-style imports. Started by whisper_wu.py; protocol in _worker.py. argv[1] is a model dir holding final.pt,
train.yaml and units.txt (wenet.cli.model.load_model's layout).
"""
import os
import sys

# Protocol replies go to a private copy of stdout; fd 1 itself is pointed at stderr *before* any import, so nothing a
# library prints (WeNet prints at import time, subprocesses inherit fd 1) can land in the reply stream.
out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)

import json  # noqa: E402
from typing import Optional, Union

import torch
import torch.nn.modules.conv as _conv
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

# WeNet's squeezeformer/conv2d.py imports these from torch.nn.modules.conv, which newer torch no longer re-exports
# (shanghainese-transcription patches the file instead). init_model imports it even for Whisper models.
for k, v in {"Union": Union, "Optional": Optional, "Tensor": torch.Tensor, "_size_2_t": _size_2_t, "_pair": _pair}.items():
    if not hasattr(_conv, k):
        setattr(_conv, k, v)

from wenet.cli.model import load_model  # noqa: E402

model = load_model(sys.argv[1], device="cpu")  # CPU, as in shanghainese-transcription; MPS untested
print(json.dumps({"ready": "cpu"}), file=out, flush=True)
for line in sys.stdin:
    try:
        with torch.inference_mode():
            text = model.transcribe(line.strip()).text or ""
        print(json.dumps({"text": text}, ensure_ascii=False), file=out, flush=True)
    except Exception as e:  # report and keep serving
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=out, flush=True)
