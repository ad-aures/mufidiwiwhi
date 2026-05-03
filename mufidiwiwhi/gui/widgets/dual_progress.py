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

"""QProgressBar that paints its percentage text in two colors so
it stays readable on both the empty-track side and the filled-
chunk side.

Qt's default progress bar paints the text once in a single color,
which means there's no good single value: dark text disappears
over the accent-blue chunk and white text disappears over the
white track. This subclass disables the default text painting and
overlays the same text twice in `paintEvent`, clipping the white
copy to the chunk rect and the dark copy to the non-chunk rect.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QProgressBar, QWidget


_TEXT_OVER_CHUNK = QColor("#ffffff")
_TEXT_OVER_TRACK = QColor("#1e1e1e")


class DualColorProgressBar(QProgressBar):
    """Progress bar with bicolor percentage text."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Suppress Qt's built-in text drawing; we render it ourselves
        # in paintEvent so we can split it across the chunk boundary.
        self.setTextVisible(False)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        rect = self.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        text = self.text() or self._default_text()
        if not text:
            return
        chunk_w = self._chunk_width(rect.width())
        chunk_rect = QRect(rect.left(), rect.top(), chunk_w, rect.height())
        track_rect = QRect(
            rect.left() + chunk_w,
            rect.top(),
            rect.width() - chunk_w,
            rect.height(),
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self.font())
        # White over the filled chunk.
        painter.save()
        painter.setClipRect(chunk_rect)
        painter.setPen(_TEXT_OVER_CHUNK)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        painter.restore()
        # Dark over the empty track.
        painter.save()
        painter.setClipRect(track_rect)
        painter.setPen(_TEXT_OVER_TRACK)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        painter.restore()
        painter.end()

    def _chunk_width(self, total_width: int) -> int:
        lo, hi = self.minimum(), self.maximum()
        if hi <= lo:
            return 0
        val = max(lo, min(hi, self.value()))
        frac = (val - lo) / (hi - lo)
        return int(round(total_width * frac))

    def _default_text(self) -> str:
        # QProgressBar.text() returns '' when textVisible was set
        # off in some Qt versions, so reproduce the default
        # 'NN%' format ourselves. Clamp to [min, max] so the value's
        # default of -1 (set when neither setValue nor setRange has
        # been called yet) doesn't render as "-1%".
        lo, hi = self.minimum(), self.maximum()
        if hi <= lo:
            return ""
        val = max(lo, min(hi, self.value()))
        pct = int(round(((val - lo) / (hi - lo)) * 100))
        return f"{pct}%"
