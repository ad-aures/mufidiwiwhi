# PyInstaller spec for the single-file Linux build.
#
#   pyinstaller packaging/mufidiwiwhi-gui-onefile.spec --noconfirm
#
# Produces a single executable in dist/mufidiwiwhi-gui. First launch
# is slower because PyInstaller unpacks the bundle to a temp dir;
# subsequent launches reuse the unpacked tree.
#
# Whisper model weights are NOT bundled (they'd add gigabytes); they
# download on first use to the user's huggingface cache (or to
# whatever Settings.model_dir points at).
#
# Hunspell .aff/.dic files are NOT bundled either; the app picks them
# up from /usr/share/hunspell at runtime.

# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# `SPECPATH` is the directory containing this spec file; the
# project root is its parent. PyInstaller resolves every relative
# path in `Analysis()` from the spec's own directory, so we have
# to anchor explicitly here or it would look for
# `packaging/mufidiwiwhi/gui/app.py` (which doesn't exist).
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# Modules PyInstaller's static analysis can't always see because
# they're loaded lazily (faster-whisper backends, our optional
# Hunspell suggester, the system-metrics probes).
HIDDEN_IMPORTS = [
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "av",
    "phonetic_fr",
    "metaphone",
    "jellyfish",
    "spylls.hunspell",
    "psutil",
    "pynvml",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtSvg",
]

# Bundle the resource directory verbatim so QPixmap / QSvgRenderer
# can find every icon, the chevron, and the tinted SVG payloads
# at runtime. The destination path mirrors the package layout so
# `importlib.resources` resolves the same way as in dev.
DATAS = [
    (os.path.join(PROJECT_ROOT, "mufidiwiwhi", "resources"),
     "mufidiwiwhi/resources"),
]

# Heavy modules we DELIBERATELY don't ship in the bundle.
#
# faster-whisper uses ctranslate2 (which has its own bundled CUDA
# libs) for inference, so torch and the NVIDIA wheel ecosystem
# would just be dead weight: ~4 GB of dead weight, in fact.
#
# CUDA detection in the bundled binary uses pynvml (already shipped
# for the metrics strip) instead of torch.cuda.is_available(), so
# excluding torch costs us nothing at runtime.
EXCLUDES = [
    # Torch and friends.
    "torch",
    "torchaudio",
    "torchvision",
    "tensorflow",
    "tensorboard",
    "sympy",
    "networkx",
    # NVIDIA wheel CUDA libraries (only torch loads these; ctranslate2
    # uses the system / its own bundled CUDA).
    "nvidia.cuda_cupti",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.cufile",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.cusparselt",
    "nvidia.nccl",
    "nvidia.nvjitlink",
    "nvidia.nvshmem",
    "nvidia.nvtx",
    # JIT / compile stack pulled in by torch.
    "triton",
    "numba",
    "llvmlite",
    "tensorboard_data_server",
]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="mufidiwiwhi-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
