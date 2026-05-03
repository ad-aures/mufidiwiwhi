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

"""Drag-and-drop helper for audio file URLs."""

from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import QMimeData, QUrl


AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".opus",
    ".aac",
    ".webm",
    ".aup3",
)


def is_audio_url(url: QUrl) -> bool:
    if not url.isLocalFile():
        return False
    path = url.toLocalFile().lower()
    return any(path.endswith(ext) for ext in AUDIO_EXTENSIONS)


def mime_audio_paths(mime: QMimeData) -> list[str]:
    if not mime.hasUrls():
        return []
    return [u.toLocalFile() for u in mime.urls() if is_audio_url(u)]


def filter_audio_paths(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if any(p.lower().endswith(e) for e in AUDIO_EXTENSIONS)]
