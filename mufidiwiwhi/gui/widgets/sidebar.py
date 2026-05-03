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

"""Vertical navigation sidebar (libadwaita / GNOME Settings style).

Top to bottom:

  * brand header (icon + name + subtitle);
  * 1 px hairline divider;
  * `QListWidget` of nav rows (Settings / Project / Run);
  * stretch (pushes the search panel below to the bottom);
  * search panel — only shown when the Run tab is active. Drives
    the live log search on the Run page.

Public API matches what `MainWindow` already calls plus a few
search helpers:

  * `add_segment(label, tooltip="", icon_svg="") -> int`
  * `set_segment_enabled(index, enabled)`
  * `set_segment_tooltip(index, tooltip)`
  * `set_current(index)`
  * `set_search_visible(bool)`
  * `set_match_count(current, total)`
  * `search_input` — `_SearchLineEdit` widget exposed for wiring.
  * signals `currentChanged(int)`
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import __version__
from ...resources import icon_path
from .icons import render_svg_to_pixmap, tinted_icon, tinted_pixmap


def _svg_path_pixmap(path: str, size: int) -> QPixmap:
    """Untinted file-path SVG → DPR-aware pixmap. Wraps
    `render_svg_to_pixmap` for the brand icon path-on-disk case."""
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return QPixmap()
    return render_svg_to_pixmap(renderer, size)


SIDEBAR_WIDTH = 232
_NAV_ICON_PX = 16


class _SearchLineEdit(QLineEdit):
    """QLineEdit that distinguishes Enter from Shift+Enter and
    Escape, so the parent can drive next/prev/clear from a single
    field. Shift+Enter and Escape don't propagate to the default
    line-edit handlers.
    """

    nextRequested = pyqtSignal()
    prevRequested = pyqtSignal()
    escapePressed = pyqtSignal()

    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.prevRequested.emit()
            else:
                self.nextRequested.emit()
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.escapePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class Sidebar(QFrame):
    """Left-hand navigation column."""

    currentChanged = pyqtSignal(int)

    def __init__(
        self, app_name: str, subtitle: str, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header(app_name, subtitle))
        layout.addWidget(self._make_divider())

        self._list = QListWidget()
        self._list.setObjectName("sidebarList")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setUniformItemSizes(True)
        self._list.setIconSize(QSize(_NAV_ICON_PX, _NAV_ICON_PX))
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, stretch=1)

        self._search_panel = self._build_search_panel()
        self._search_panel.setVisible(False)
        layout.addWidget(self._search_panel)

        version_label = QLabel(f"v{__version__}")
        version_label.setObjectName("sidebarVersion")
        version_label.setContentsMargins(16, 6, 16, 10)
        version_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(version_label)

    # ----- header --------------------------------------------------------
    def _build_header(self, app_name: str, subtitle: str) -> QWidget:
        header = QFrame()
        header.setObjectName("sidebarHeader")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 14, 16, 14)
        h_layout.setSpacing(12)

        icon_lbl = QLabel()
        ipath = icon_path()
        if ipath:
            pix = _svg_path_pixmap(ipath, 32)
            if not pix.isNull():
                icon_lbl.setPixmap(pix)
        icon_lbl.setFixedSize(32, 32)
        h_layout.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)

        text_block = QVBoxLayout()
        text_block.setContentsMargins(0, 0, 0, 0)
        text_block.setSpacing(2)
        name_lbl = QLabel(app_name)
        name_lbl.setObjectName("sidebarAppName")
        text_block.addWidget(name_lbl)
        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("sidebarSubtitle")
        sub_lbl.setWordWrap(True)
        text_block.addWidget(sub_lbl)
        h_layout.addLayout(text_block, stretch=1)
        return header

    @staticmethod
    def _make_divider() -> QFrame:
        d = QFrame()
        d.setObjectName("sidebarDivider")
        d.setFixedHeight(1)
        return d

    # ----- search panel --------------------------------------------------
    def _build_search_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebarBottomSearch")
        p_layout = QVBoxLayout(panel)
        p_layout.setContentsMargins(12, 8, 12, 12)
        p_layout.setSpacing(4)

        self.search_input = _SearchLineEdit()
        self.search_input.setObjectName("sidebarSearchInput")
        self.search_input.setPlaceholderText(self.tr("Search log..."))
        self.search_input.setClearButtonEnabled(True)
        # Leading magnifier icon, baked into the line edit itself
        # (avoids any wrapping QFrame that would have to track focus
        # state separately).
        pix = tinted_pixmap("magnifer-svgrepo-com.svg", "#5e5e5e", 16)
        if pix is not None:
            search_action = QAction(QIcon(pix), "", self.search_input)
            self.search_input.addAction(
                search_action, QLineEdit.ActionPosition.LeadingPosition
            )
        # Trailing "find next" button: clicking it is the same as
        # pressing Enter in the field.
        next_pix = tinted_pixmap("square-alt-arrow-down-svgrepo-com.svg", "#5e5e5e", 16)
        if next_pix is not None:
            next_action = QAction(QIcon(next_pix), "", self.search_input)
            next_action.setToolTip(self.tr("Find next"))
            next_action.triggered.connect(self.search_input.nextRequested.emit)
            self.search_input.addAction(
                next_action, QLineEdit.ActionPosition.TrailingPosition
            )
        p_layout.addWidget(self.search_input)

        self.match_label = QLabel("")
        self.match_label.setObjectName("sidebarSearchCount")
        self.match_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        p_layout.addWidget(self.match_label)
        return panel

    # ----- public API ----------------------------------------------------
    def add_segment(
        self, label: str, tooltip: str = "", icon_svg: str = ""
    ) -> int:
        item = QListWidgetItem(label)
        if tooltip:
            item.setToolTip(tooltip)
        if icon_svg:
            item.setIcon(tinted_icon(icon_svg, "#1e1e1e", _NAV_ICON_PX))
        self._list.addItem(item)
        idx = self._list.count() - 1
        if idx == 0:
            self._list.setCurrentRow(0)
        return idx

    def set_segment_enabled(self, index: int, enabled: bool) -> None:
        item = self._list.item(index)
        if item is None:
            return
        flags = item.flags()
        if enabled:
            item.setFlags(
                flags
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
        else:
            item.setFlags(
                flags
                & ~Qt.ItemFlag.ItemIsEnabled
                & ~Qt.ItemFlag.ItemIsSelectable
            )

    def set_segment_tooltip(self, index: int, tooltip: str) -> None:
        item = self._list.item(index)
        if item is not None:
            item.setToolTip(tooltip)

    def set_current(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    def current(self) -> int:
        return self._list.currentRow()

    def set_search_visible(self, visible: bool) -> None:
        self._search_panel.setVisible(visible)

    def set_match_count(self, current: int, total: int) -> None:
        """Update the small "X/Y" label below the search field.
        When the query is empty the label clears completely; when
        there are zero matches we tag it with a 'warn' tone.
        """
        if not self.search_input.text():
            self.match_label.setText("")
            self._set_match_tone("muted")
            return
        if total == 0:
            self.match_label.setText(self.tr("0/0"))
            self._set_match_tone("warn")
        else:
            self.match_label.setText(f"{current}/{total}")
            self._set_match_tone("muted")

    def _set_match_tone(self, tone: str) -> None:
        self.match_label.setProperty("tone", tone)
        self.match_label.style().unpolish(self.match_label)
        self.match_label.style().polish(self.match_label)

    # ----- internal ------------------------------------------------------
    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.currentChanged.emit(row)
