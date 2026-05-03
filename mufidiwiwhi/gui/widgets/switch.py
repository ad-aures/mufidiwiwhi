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

"""GNOME 49 / libadwaita pill switch.

A `QAbstractButton` (`setCheckable(True)`) that paints itself as a
~48 by 26 px pill: grey track when off, accent track when on, with
a 22 px white knob that sits 2 px inset from the edge. Hover
darkens the track slightly. Disabled state is rendered at half
opacity. Click toggles `isChecked`; the standard `toggled(bool)`
signal carries the change.

The widget is purely painted (no QSS) so it has the same look on
every platform.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QAbstractButton, QSizePolicy, QWidget


_TRACK_OFF = QColor("#cfcfcf")
_TRACK_OFF_HOVER = QColor("#c0c0c0")
_TRACK_ON = QColor("#006699")  # ACCENT
_TRACK_ON_HOVER = QColor("#005580")
_KNOB = QColor("#ffffff")
_DISABLED_TRACK = QColor("#dddddd")
_DISABLED_KNOB = QColor("#f1f1f1")


class Switch(QAbstractButton):
    """Pill-shaped on/off switch."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(48, 26)
        self._hovered = False

    # ----- size --------------------------------------------------------
    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(48, 26)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(48, 26)

    # ----- hover / press -----------------------------------------------
    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    # ----- paint -------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect())
        radius = rect.height() / 2.0

        # Track color depends on state.
        if not self.isEnabled():
            track = _DISABLED_TRACK
            knob = _DISABLED_KNOB
            painter.setOpacity(0.65)
        elif self.isChecked():
            track = _TRACK_ON_HOVER if self._hovered else _TRACK_ON
            knob = _KNOB
        else:
            track = _TRACK_OFF_HOVER if self._hovered else _TRACK_OFF
            knob = _KNOB

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(rect, radius, radius)

        # Knob.
        knob_inset = 2.0
        knob_diam = rect.height() - 2 * knob_inset
        if self.isChecked():
            knob_x = rect.right() - knob_inset - knob_diam
        else:
            knob_x = rect.left() + knob_inset
        knob_rect = QRectF(knob_x, rect.top() + knob_inset, knob_diam, knob_diam)
        painter.setBrush(knob)
        painter.drawEllipse(knob_rect)
        painter.end()

    # Default keyboard activation: space / enter via QAbstractButton.
