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

"""Three round window-control buttons (minimize / maximize / close).

GNOME 49 / Adwaita style: identical pale-grey circles at all
times. Each holds a small thin-stroked symbolic icon (a line, a
square, two overlapping squares for restore, an x). The minimize
and maximize buttons darken slightly on hover; the close button
turns red with a white icon on hover.

The three buttons are 24x24 px circles (border-radius 12 px). The
maximize button swaps to a "restore" icon when the window is
maximized; `WindowControls.set_maximized()` is called from the
`MainWindow.changeEvent` handler.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from .icons import tinted_icon


_ICON_NORMAL = "#2e2e2e"
_ICON_HOVER = "#ffffff"
_ICON_PX = 12


class _CtrlButton(QPushButton):
    """24x24 round button. If `icon_hover_color` is given, the icon
    is replaced on hover (used for the close button so its glyph
    flips to white over the red background).
    """

    def __init__(
        self,
        svg_name: str,
        role: str,
        tooltip: str,
        parent: Optional[QWidget] = None,
        icon_hover_color: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "windowControlBtn")
        self.setProperty("role", role)
        self.setFixedSize(24, 24)
        self.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if tooltip:
            self.setToolTip(tooltip)
        self._svg_name = svg_name
        self._icon_normal = self._load_icon(svg_name, _ICON_NORMAL)
        self._icon_hover: Optional[QIcon] = (
            self._load_icon(svg_name, icon_hover_color)
            if icon_hover_color is not None
            else None
        )
        self.setIcon(self._icon_normal)

    @staticmethod
    def _load_icon(name: str, color: str) -> QIcon:
        return tinted_icon(name, color, _ICON_PX)

    def set_svg(self, svg_name: str) -> None:
        """Swap the underlying SVG (used to flip maximize↔restore)."""
        self._svg_name = svg_name
        self._icon_normal = self._load_icon(svg_name, _ICON_NORMAL)
        # Don't touch the hover icon: only the close button uses it,
        # and the close button never swaps SVG.
        self.setIcon(self._icon_normal)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        if self._icon_hover is not None:
            self.setIcon(self._icon_hover)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._icon_hover is not None:
            self.setIcon(self._icon_normal)
        super().leaveEvent(event)


class WindowControls(QFrame):
    """Three round buttons in a horizontal strip."""

    minimizeRequested = pyqtSignal()
    maximizeToggleRequested = pyqtSignal()
    closeRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("windowControls")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self._min_btn = _CtrlButton(
            "window-minimize.svg",
            "min",
            self.tr("Minimize"),
            icon_hover_color=_ICON_HOVER,
        )
        self._max_btn = _CtrlButton(
            "window-maximize.svg",
            "max",
            self.tr("Maximize"),
            icon_hover_color=_ICON_HOVER,
        )
        self._close_btn = _CtrlButton(
            "window-close.svg",
            "close",
            self.tr("Close"),
            icon_hover_color=_ICON_HOVER,
        )
        for b in (self._min_btn, self._max_btn, self._close_btn):
            h.addWidget(b)

        self._min_btn.clicked.connect(self.minimizeRequested.emit)
        self._max_btn.clicked.connect(self.maximizeToggleRequested.emit)
        self._close_btn.clicked.connect(self.closeRequested.emit)

    def set_maximized(self, maximized: bool) -> None:
        # Keep the maximize glyph at all times; only the tooltip
        # tracks the actual action that will fire on click.
        self._max_btn.setToolTip(
            self.tr("Restore") if maximized else self.tr("Maximize")
        )
