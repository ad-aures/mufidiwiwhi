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

"""Adwaita / GNOME-49 inspired primitives: CardFrame, ActionRow, EntryRow.

A `CardFrame` is a single rounded white panel that contains a vertical
stack of rows. Rows fill the card edge-to-edge and are separated by a
1 px hairline that's inset 16 px on both sides.

Two row variants:

* `ActionRow`: title + optional subtitle on the left, a control widget
  on the right. Use for switches, dropdowns, action buttons.
* `EntryRow`: floating small label on top, editable value below. Use
  for `QLineEdit` / `QSpinBox` / `QComboBox`. The whole row gets a
  2 px focus outline when the contained widget gains focus.

The QSS in `style.py` styles these widgets via the `class` property
(e.g. `QFrame[class~="card"]`). We use the `~=` selector to allow
multiple class tokens on a single widget if needed later.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# CardFrame
# ---------------------------------------------------------------------------


class CardFrame(QFrame):
    """A rounded white panel that holds a stack of rows separated by
    hairline dividers. Rows are added with `add_row`.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._inner = layout

    def add_row(self, row: QWidget) -> None:
        if self._inner.count() > 0:
            self._inner.addWidget(_row_divider())
        self._inner.addWidget(row)

    def add_widget(self, widget: QWidget) -> None:
        """Add an arbitrary widget WITHOUT a divider above it. Useful
        for header bands that intentionally lack separators.
        """
        self._inner.addWidget(widget)


def _row_divider() -> QFrame:
    wrap = QFrame()
    wrap.setProperty("class", "rowDividerWrap")
    wrap.setFixedHeight(1)
    layout = QHBoxLayout(wrap)
    layout.setContentsMargins(16, 0, 16, 0)
    layout.setSpacing(0)
    line = QFrame()
    line.setProperty("class", "rowDivider")
    line.setFixedHeight(1)
    layout.addWidget(line)
    return wrap


# ---------------------------------------------------------------------------
# Row title block helpers
# ---------------------------------------------------------------------------


def _make_title_block(title: str, subtitle: Optional[str] = None) -> QWidget:
    box = QWidget()
    box.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    title_label = QLabel(title)
    title_label.setProperty("class", "rowTitle")
    title_label.setWordWrap(False)
    layout.addWidget(title_label)
    if subtitle:
        sub_label = QLabel(subtitle)
        sub_label.setProperty("class", "rowSubtitle")
        sub_label.setWordWrap(True)
        layout.addWidget(sub_label)
    return box


# ---------------------------------------------------------------------------
# ActionRow: title + subtitle on left, control(s) on right.
# ---------------------------------------------------------------------------


class ActionRow(QFrame):
    """A row with a title (and optional subtitle) on the left and one
    or more control widgets on the right.
    """

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        controls: Optional[list[QWidget]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "row")
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 12, 16, 12)
        h.setSpacing(12)
        h.addWidget(_make_title_block(title, subtitle), stretch=1)
        if controls:
            for w in controls:
                h.addWidget(w)


# ---------------------------------------------------------------------------
# EntryRow: floating label on top, editable value below.
# ---------------------------------------------------------------------------


class EntryRow(QFrame):
    """A row whose label sits as small floating text above an
    editable value (line edit / spin box / combo). The whole row
    shows a 2 px accent-colored outline while the inner widget has
    focus.

    `extra_buttons` are placed to the right of the value (e.g.
    "Browse...", "Edit...").
    """

    def __init__(
        self,
        label: str,
        value_widget: QWidget,
        extra_buttons: Optional[list[QWidget]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "row")
        self.setProperty("entry", True)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 8, 16, 8)
        outer.setSpacing(12)

        text_block = QVBoxLayout()
        text_block.setContentsMargins(0, 0, 0, 0)
        text_block.setSpacing(0)
        label_widget = QLabel(label)
        label_widget.setProperty("class", "floatingLabel")
        text_block.addWidget(label_widget)

        # Strip any inherited frame so the value visually merges with
        # the row card.
        value_widget.setProperty("class", "borderlessInput")
        if isinstance(value_widget, QLineEdit):
            value_widget.setFrame(False)
        text_block.addWidget(value_widget)
        outer.addLayout(text_block, stretch=1)

        if extra_buttons:
            for b in extra_buttons:
                outer.addWidget(b)

        self._value = value_widget
        self._value.installEventFilter(self)

    def value_widget(self) -> QWidget:
        return self._value

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._value:
            if event.type() == QEvent.Type.FocusIn:
                self._set_focused(True)
            elif event.type() == QEvent.Type.FocusOut:
                self._set_focused(False)
        return False

    def _set_focused(self, focused: bool) -> None:
        self.setProperty("focused", focused)
        # Force a re-polish so the QSS rule keyed on the property
        # takes effect immediately.
        self.style().unpolish(self)
        self.style().polish(self)


# ---------------------------------------------------------------------------
# Section header (sits ABOVE a card, like the libadwaita PreferencesGroup
# title). Small, semi-bold, slightly muted.
# ---------------------------------------------------------------------------


class SectionHeader(QLabel):
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setProperty("class", "sectionHeader")
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(font.pointSize())  # keep size; weight via QSS
        self.setFont(font)
