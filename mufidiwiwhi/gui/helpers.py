# Mufidiwiwhi - multi-file diarisation transcription with Whisper.
# (C) 2026 Ad Aures · Benjamin Bellamy <benjamin@podlibre.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Sensible-defaults and runtime probes for the GUI.

Kept in a single module so the Setup page (and tests) do not duplicate
the logic. None of these probes raise; they return safe fallbacks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardware / environment detection
# ---------------------------------------------------------------------------


def detect_cuda() -> tuple[bool, str]:
    """Return (cuda_available, label) for display in the GUI.

    Uses pynvml first (already a hard dependency for the metrics
    strip and far smaller than torch) and falls back to torch only
    if pynvml is missing. The PyInstaller bundle ships pynvml but
    NOT torch, so the binary stays slim.
    """
    # pynvml path
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return False, "CPU only (no CUDA)"
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                raw = pynvml.nvmlDeviceGetName(handle)
                name = raw.decode("utf-8", "replace") if isinstance(
                    raw, bytes
                ) else str(raw)
            except Exception:
                name = "CUDA"
            vram_gb = detect_vram_gb()
            label = (
                f"CUDA: {name} ({vram_gb:.1f} GB)"
                if vram_gb
                else f"CUDA: {name}"
            )
            return True, label
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    # torch fallback (only present when running from source / dev venv)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA"
            vram_gb = detect_vram_gb()
            if vram_gb:
                return True, f"CUDA: {name} ({vram_gb:.1f} GB)"
            return True, f"CUDA: {name}"
    except Exception:
        pass
    return False, "CPU only (no CUDA)"


def detect_vram_gb() -> float:
    """Return total VRAM of GPU 0 in GB. 0.0 if no CUDA."""
    # pynvml path first (no torch dependency).
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() <= 0:
                return 0.0
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(mem.total) / (1024 ** 3)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        bytes_total = torch.cuda.get_device_properties(0).total_memory
        return bytes_total / (1024 ** 3)
    except Exception:
        return 0.0


def default_device() -> str:
    """Return 'cuda' when usable, otherwise 'cpu'. Never returns 'auto'.

    The CLI accepts 'auto' but the GUI prefers to make the choice
    explicit so the user sees what will run.
    """
    cuda_available, _ = detect_cuda()
    return "cuda" if cuda_available else "cpu"


# Approximate VRAM (GB) needed to run each Whisper model in float16. CPU
# inference uses RAM and is not bounded by these. The numbers are
# conservative to leave headroom for the audio buffer.
_MODEL_VRAM_GB: dict[str, float] = {
    "tiny": 1.0,
    "tiny.en": 1.0,
    "base": 1.5,
    "base.en": 1.5,
    "small": 2.5,
    "small.en": 2.5,
    "medium": 5.5,
    "medium.en": 5.5,
    "large-v1": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.0,
    "large": 10.0,
    "distil-small.en": 2.0,
    "distil-medium.en": 4.0,
    "distil-large-v2": 6.5,
    "distil-large-v3": 6.5,
}


def model_vram_gb(model_name: str) -> float:
    """Approximate VRAM needed by a Whisper model, in GB. 0 if unknown."""
    return _MODEL_VRAM_GB.get(model_name.strip(), 0.0)


def model_fits_on_gpu(model_name: str, vram_gb: float) -> bool:
    """True if the model is expected to load on a GPU with `vram_gb` of VRAM.

    Reserves 1 GB of headroom for activations and other buffers.
    """
    needed = model_vram_gb(model_name)
    if needed <= 0:
        return True  # unknown model: do not block
    return vram_gb - 1.0 >= needed


_COMPUTE_TYPES_GPU = ("auto", "float16", "int8_float16", "int8", "float32")
_COMPUTE_TYPES_CPU = ("auto", "int8", "float32")


def compute_types_for_device(device: str) -> tuple[str, ...]:
    """Return the meaningful compute types for `device`."""
    if device == "cuda":
        return _COMPUTE_TYPES_GPU
    return _COMPUTE_TYPES_CPU


def recommend_compute_type(device: str) -> str:
    """Best compute_type for `device`.

    On GPU, `float16` is the standard fast path and `int8_float16`
    saves memory at a small accuracy cost. On CPU, `int8` is much
    faster than `float32`.
    """
    if device == "cuda":
        return "float16"
    return "int8"


def resolve_compute_type(value: str, device: str) -> str:
    """Translate the GUI's 'auto' to a real faster-whisper value."""
    if value == "auto" or not value:
        return recommend_compute_type(device)
    return value


def recommend_whisper_model(device: str, vram_gb: float) -> str:
    """Return the recommended Whisper model for a given device + VRAM.

    Heuristics:
      - GPU with >= 11 GB VRAM: large-v3 (best accuracy).
      - GPU with 6 to 11 GB: medium (good accuracy / fits comfortably).
      - GPU with 3 to 6 GB: small.
      - GPU with < 3 GB: base.
      - CPU: small (medium is slow on CPU; large is impractical).
    """
    if device != "cuda" or vram_gb <= 0:
        return "small"
    if vram_gb >= 11:
        return "large-v3"
    if vram_gb >= 6:
        return "medium"
    if vram_gb >= 3:
        return "small"
    return "base"


# ---------------------------------------------------------------------------
# Model cache directory + downloaded models
# ---------------------------------------------------------------------------


def default_model_dir() -> str:
    """Return the default faster-whisper / huggingface cache directory.

    faster-whisper resolves `download_root=None` to the platform cache
    dir; we mirror its layout so the GUI shows the same place the user
    would otherwise have to find by reading the source.
    """
    env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return os.path.join(env, "hub") if env.endswith(".cache") else env
    home = Path.home()
    return str(home / ".cache" / "huggingface" / "hub")


_MODEL_DIRNAME_RE = re.compile(
    r"^models--(?:[^-]+--)?(?:Systran|guillaumekln|deepdml)--"
    r"faster-whisper-(?P<name>[A-Za-z0-9._-]+)$"
)


def list_downloaded_models(model_dir: str | None = None) -> list[str]:
    """List the faster-whisper model names already cached on disk.

    Returns the short model names ("medium", "large-v3") sorted, with
    duplicates removed. Empty list if the directory does not exist.
    """
    base = model_dir or default_model_dir()
    if not base or not os.path.isdir(base):
        return []
    found: set[str] = set()
    try:
        for entry in os.listdir(base):
            m = _MODEL_DIRNAME_RE.match(entry)
            if m:
                found.add(m.group("name"))
    except OSError:
        return []
    return sorted(found)


