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

"""GNOME-49-style segmented "view switcher" control.

Three pill segments inside a single rounded background. Drives a
QStackedWidget via a `currentChanged(int)` signal so callers don't
need to know about QTabWidget at all.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
)


class ViewSwitcher(QFrame):
    """Horizontal segmented control. Each call to `add_segment` appends
    a labelled pill; the active segment is highlighted with the accent
    colour. Use `set_segment_enabled` to grey out a segment without
    removing it.
    """

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viewSwitcher")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idToggled.connect(self._on_id_toggled)
        self._buttons: list[QPushButton] = []

    def add_segment(self, label: str, tooltip: str = "") -> int:
        idx = len(self._buttons)
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setProperty("segment", True)
        if tooltip:
            btn.setToolTip(tooltip)
        if idx == 0:
            btn.setChecked(True)
        self._row.addWidget(btn)
        self._group.addButton(btn, idx)
        self._buttons.append(btn)
        return idx

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def current(self) -> int:
        for i, b in enumerate(self._buttons):
            if b.isChecked():
                return i
        return -1

    def set_segment_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(enabled)

    def set_segment_tooltip(self, index: int, tooltip: str) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setToolTip(tooltip)

    def _on_id_toggled(self, idx: int, checked: bool) -> None:
        if checked:
            self.currentChanged.emit(idx)
