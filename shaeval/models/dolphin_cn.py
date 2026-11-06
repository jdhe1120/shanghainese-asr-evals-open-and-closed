"""DataoceanAI Dolphin-CN-Dialect small.cn (0.4 B, May 2026), forced to Shanghainese: lang_sym zh, region_sym SHANGHAI
(languages.md lists both zh-SHANGHAI 上海话 and the broader zh-WU 吴语).

The PyPI package is `dataoceanai-dolphin`; `pip install dolphin` (what the model card says) installs an unrelated InSAR
package. Its own download is broken: its model table asks ModelScope for "DataoceanAI/dolphi-cn-dialect-small" (typo),
so scripts/fetch_models.py fetches DataoceanAI/dolphin-cn-dialect-small into models/dolphin-cn-dialect-small
(small.cn.pt sha256 matches the package's table, so load_model accepts it without re-downloading). DOLPHIN_DIR in .env
overrides.

Dolphin's feature extraction applies its training config's fbank dither (0.1) at inference too, i.e. random noise.
Kept as the vendor ships it, but torch is re-seeded per clip so a re-run gives the same text.
"""
import re
from pathlib import Path

from ._env import ROOT, env
from ._qwen import device

MODEL = "small.cn"
SEED = 0
_model = None


def transcribe(wav_path: str) -> str:
    global _model
    import dolphin
    import torch

    if _model is None:
        model_dir = Path(env().get("DOLPHIN_DIR") or ROOT / "models" / "dolphin-cn-dialect-small")
        # MPS untested in dolphin, so CPU on a Mac
        _model = dolphin.load_model(MODEL, model_dir=model_dir, device="cpu" if device() == "mps" else device())
    torch.manual_seed(SEED)
    text = dolphin.transcribe(_model, wav_path, lang_sym="zh", region_sym="SHANGHAI").text
    return re.sub(r"<[^>]*>", "", text).strip()  # drop the <zh><SHANGHAI><asr><notimestamp> prefix
