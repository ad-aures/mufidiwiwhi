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

from setuptools import find_packages, setup

setup(
    name="mufidiwiwhi",
    version="2.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mufidiwiwhi.resources": ["*.svg", "*.png", "*.md"],
    },
    license="GPL-3.0-or-later",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    python_requires=">=3.10",
    install_requires=[
        "faster-whisper>=1.0.3",
        "numpy>=1.24",
        "scipy>=1.10",
        "aup3>=1.0.1",
        "phonetic-fr>=1.0",
        "metaphone>=0.6",
        "jellyfish>=1.0",
        "spylls>=0.1.7",
        "psutil>=5.9",
        "nvidia-ml-py>=12",
        "PyQt6>=6.7",
    ],
    extras_require={
        "dev": [
            "pytest>=8",
            "pyinstaller>=6",
            "pydub>=0.25",
        ],
    },
    entry_points={
        "console_scripts": [
            "mufidiwiwhi = mufidiwiwhi.cli:main",
            "mufidiwiwhi-gui = mufidiwiwhi.gui.app:main",
        ]
    },
)
