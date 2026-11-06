"""Client for a model that runs in its own `uv run --isolated` process (its dependencies clash with the main env).

The worker script prints one JSON line when ready ({"ready": <device>}), then reads one wav path per stdin line and
answers with one JSON line ({"text": ...} or {"error": ...}). The process stays up for the whole run.
"""
import atexit
import json
import subprocess
from pathlib import Path


_all: list["Worker"] = []


def stop_all() -> None:
    """End every worker, freeing its memory; one that is needed again restarts on its next clip."""
    for w in _all:
        if w.proc is not None and w.proc.poll() is None:
            w.proc.terminate()
            w.proc.wait()


class Worker:
    def __init__(self, name: str, cmd: list[str]):
        self.name, self.cmd, self.proc = name, cmd, None
        _all.append(self)

    def _start(self) -> subprocess.Popen:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         encoding="utf-8")
            atexit.register(self.proc.terminate)
            ready = self.proc.stdout.readline()
            if not ready:
                raise RuntimeError(f"{self.name} worker exited during start-up (see stderr above)")
            print(f"[{self.name}] worker ready on {json.loads(ready)['ready']}", flush=True)
        return self.proc

    def transcribe(self, wav_path: str) -> str:
        p = self._start()
        p.stdin.write(str(Path(wav_path).resolve()) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} worker died (see stderr above)")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["text"].strip()
