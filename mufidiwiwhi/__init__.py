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

"""Mufidiwiwhi: multi-file diarisation transcription with Whisper."""

__version__ = "2.1.0"


def _preload_cublas() -> None:
    # ctranslate2 dlopens libcublas.so.12 lazily and relies on the
    # dynamic loader. The nvidia-cublas-cu12 wheel ships the .so files
    # but does not register them on the loader path, so without torch
    # (whose RPATH does it as a side-effect) the dlopen fails. Load
    # them ourselves here so ctranslate2's later dlopen succeeds.
    import ctypes
    import os
    import sys

    if sys.platform != "linux":
        return
    try:
        import nvidia.cublas as _cublas_pkg
    except ImportError:
        return
    pkg_path = getattr(_cublas_pkg, "__path__", None)
    if not pkg_path:
        return
    lib_dir = os.path.join(next(iter(pkg_path)), "lib")
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cublas()
del _preload_cublas


from .core import (
    Cancelled,
    CorrectionConfig,
    RunConfig,
    SpeakerInput,
    run_pipeline,
)

__all__ = [
    "Cancelled",
    "CorrectionConfig",
    "RunConfig",
    "SpeakerInput",
    "__version__",
    "run_pipeline",
]
