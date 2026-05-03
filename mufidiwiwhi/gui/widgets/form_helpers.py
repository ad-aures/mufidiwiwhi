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

"""Tiny helpers for building forms with tooltip-bearing labels.

`QFormLayout.addRow(str, widget)` builds a plain QLabel internally and
gives us no handle on it, so tooltips on the left-hand label do not
work out of the box. These helpers create the label ourselves so we
can attach a tooltip and apply the same tooltip to the field.

Tooltips are wrapped in a `<qt>` block so Qt treats them as rich text
and word-wraps automatically (otherwise long single-line tooltips run
off the screen).

`labelled_row` accepts either a `QWidget` or a `QLayout` as the field,
which is the cleanest way to put compound rows (line edit + Browse
button) into a form without introducing a wrapper `QWidget` that
confuses `QFormLayout`'s vertical sizing.
"""

from __future__ import annotations

import re
from typing import Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QBoxLayout,
    QFormLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QWidget,
)


Field = Union[QWidget, QLayout]


_TOOLTIP_MAX_PX = 360


def format_tooltip(text: str) -> str:
    """Wrap a tooltip in a width-constrained HTML block.

    Qt only wraps tooltip text that is recognized as rich text. We
    wrap the body in `<qt>` and break the body into paragraphs by
    sentence so the layout is always multi-line and readable.
    """
    if not text:
        return ""
    if text.lstrip().startswith("<"):
        return text  # already rich text
    # Treat blank lines or `\n\n` as paragraph breaks; single `\n`
    # becomes a soft <br/>.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    body = "<br/><br/>".join(p.replace("\n", "<br/>") for p in paragraphs)
    return (
        f"<qt><div style='max-width:{_TOOLTIP_MAX_PX}px;'>"
        f"{body}</div></qt>"
    )


def _apply_tooltip_to_field(field: Field, tooltip: str) -> None:
    """Apply `tooltip` to a field. For layouts, walk the immediate
    children and tooltip them so hovering anywhere in the row works.
    """
    if not tooltip:
        return
    if isinstance(field, QWidget):
        field.setToolTip(tooltip)
        return
    if isinstance(field, QBoxLayout):
        for i in range(field.count()):
            item = field.itemAt(i)
            w = item.widget()
            if w is not None and not w.toolTip():
                w.setToolTip(tooltip)


def labelled_row(
    form: QFormLayout,
    label_text: str,
    tooltip: str,
    field: Field,
) -> QLabel:
    """Add a row to `form` with a tooltip-bearing left-hand label."""
    lbl = _make_label(label_text, tooltip, field)
    form.addRow(lbl, field)
    return lbl


def _make_label(label_text: str, tooltip: str, field: Field) -> QLabel:
    lbl = QLabel(label_text)
    lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tooltip:
        formatted = format_tooltip(tooltip)
        lbl.setToolTip(formatted)
        _apply_tooltip_to_field(field, formatted)
    return lbl


class TwoColumnGrid:
    """Helper that lays out form rows in two side-by-side columns.

    Wraps a `QGridLayout` with column 0 / 2 for labels (right-aligned)
    and column 1 / 3 for fields (stretching). Long rows can span the
    full width by calling `add_full`.
    """

    from PyQt6.QtWidgets import QGridLayout  # noqa: E501

    def __init__(self) -> None:
        from PyQt6.QtWidgets import QGridLayout

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 0)
        self.grid.setColumnStretch(3, 1)
        self._row = 0
        self._half_pending: tuple[str, str, Field] | None = None

    def add_pair(
        self,
        left_label: str, left_tip: str, left_field: Field,
        right_label: str, right_tip: str, right_field: Field,
    ) -> None:
        self._place(self._row, 0, left_label, left_tip, left_field)
        self._place(self._row, 2, right_label, right_tip, right_field)
        self._row += 1
        self._half_pending = None

    def add_half(self, label: str, tooltip: str, field: Field) -> None:
        """Add a single short field. Pairs with the next add_half call."""
        if self._half_pending is None:
            self._half_pending = (label, tooltip, field)
            return
        # flush both halves on this row
        prev_label, prev_tip, prev_field = self._half_pending
        self._place(self._row, 0, prev_label, prev_tip, prev_field)
        self._place(self._row, 2, label, tooltip, field)
        self._row += 1
        self._half_pending = None

    def add_full(self, label: str, tooltip: str, field: Field) -> None:
        # flush a pending half on its own row first
        if self._half_pending is not None:
            prev_label, prev_tip, prev_field = self._half_pending
            self._place(self._row, 0, prev_label, prev_tip, prev_field, span=3)
            self._row += 1
            self._half_pending = None
        self._place(self._row, 0, label, tooltip, field, span=3)
        self._row += 1

    def flush(self) -> None:
        """Flush a trailing add_half so it occupies a single short row."""
        if self._half_pending is not None:
            prev_label, prev_tip, prev_field = self._half_pending
            self._place(self._row, 0, prev_label, prev_tip, prev_field)
            self._row += 1
            self._half_pending = None

    def _place(
        self,
        row: int, col: int,
        label_text: str, tooltip: str, field: Field,
        span: int = 1,
    ) -> None:
        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtWidgets import QBoxLayout

        lbl = _make_label(label_text, tooltip, field)
        self.grid.addWidget(lbl, row, col)
        if isinstance(field, QBoxLayout):
            self.grid.addLayout(field, row, col + 1, 1, span)
        else:
            self.grid.addWidget(field, row, col + 1, 1, span)
