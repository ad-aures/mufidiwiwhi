# PyInstaller spec for the single-folder Linux build.
#
#   pyinstaller packaging/mufidiwiwhi-gui-onedir.spec --noconfirm
#
# Produces dist/mufidiwiwhi-gui/ as a directory tree (faster startup
# than the onefile build, easier to debug, but more files to ship).

# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Anchor every path in `Analysis()` to the project root, NOT the
# spec's own directory (PyInstaller's default), so the build runs
# correctly from anywhere.
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

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

DATAS = [
    (os.path.join(PROJECT_ROOT, "mufidiwiwhi", "resources"),
     "mufidiwiwhi/resources"),
]

# See the onefile spec for the rationale: torch + NVIDIA wheels are
# dead weight when faster-whisper uses ctranslate2.
EXCLUDES = [
    "torch", "torchaudio", "torchvision",
    "tensorflow", "tensorboard", "sympy", "networkx",
    "nvidia.cuda_cupti", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime",
    "nvidia.cublas", "nvidia.cudnn", "nvidia.cufft", "nvidia.cufile",
    "nvidia.curand", "nvidia.cusolver", "nvidia.cusparse",
    "nvidia.cusparselt", "nvidia.nccl", "nvidia.nvjitlink",
    "nvidia.nvshmem", "nvidia.nvtx",
    "triton", "numba", "llvmlite", "tensorboard_data_server",
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
    [],
    exclude_binaries=True,
    name="mufidiwiwhi-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="mufidiwiwhi-gui",
)
