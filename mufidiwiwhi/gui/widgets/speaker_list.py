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

"""GNOME-style editable list view for the SpeakerTableModel.

Each speaker is a row with a bold title + muted file path subtitle
and two trailing icon buttons: Edit name + Remove. A final
clickable row at the bottom emits `addRequested`, which the page
wires to its file picker.

Drag-and-drop of audio files onto the card adds new rows, just
like the previous table view did.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .adwaita import CardFrame
from .audio_drop import mime_audio_paths
from .icons import IconButton, tinted_pixmap
from .speaker_table import SpeakerTableModel


class _SpeakerRow(QFrame):
    """One row inside the editable list."""

    def __init__(
        self,
        index: int,
        speaker: str,
        file_path: str,
        on_edit,
        on_remove,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "row")
        # Preferred/Preferred (rather than Preferred/Fixed): on real
        # displays the actual font metrics may differ from what was
        # available when the sizeHint was first computed, so we let
        # the row grow vertically a few px to fit instead of clipping.
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 12, 12, 12)
        h.setSpacing(12)

        text_block = QVBoxLayout()
        text_block.setContentsMargins(0, 0, 0, 0)
        text_block.setSpacing(4)
        title = QLabel(speaker)
        # `speakerTitle` is paired with `rowTitle` so the QSS overrides
        # the row-title default semibold weight back to regular for
        # this list (matches GNOME's Input Sources rows).
        title.setProperty("class", "rowTitle speakerTitle")
        text_block.addWidget(title)
        sub = QLabel(file_path)
        sub.setProperty("class", "rowSubtitle")
        sub.setToolTip(file_path)
        text_block.addWidget(sub)
        h.addLayout(text_block, stretch=1)

        edit_btn = IconButton(
            "pen-new-square-svgrepo-com.svg",
            tooltip=self.tr("Edit name"),
            size=28,
            icon_size=16,
        )
        edit_btn.clicked.connect(lambda: on_edit(index))
        h.addWidget(edit_btn)

        remove_btn = IconButton(
            "close-square-svgrepo-com.svg",
            tooltip=self.tr("Remove"),
            size=28,
            icon_size=16,
        )
        remove_btn.clicked.connect(lambda: on_remove(index))
        h.addWidget(remove_btn)


class _AddSpeakerRow(QFrame):
    """Trailing row holding the '+ Add speaker…' link and a
    'Remove all' link. Each emits its own signal."""

    addClicked = pyqtSignal()
    removeAllClicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "row")
        self.setProperty("addRow", True)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 12, 16, 12)
        h.setSpacing(8)
        h.addStretch(1)

        # "+ Add speaker audio files or Audacity project" link.
        self._add_widget = self._build_link(
            "＋",
            self.tr("Add speaker audio files or Audacity project"),
            "#006699",
        )
        self._add_widget.mouseReleaseEvent = (  # type: ignore[assignment]
            lambda _e: self.addClicked.emit()
        )
        h.addWidget(self._add_widget)

        h.addSpacing(24)

        # "Remove all" link, rendered in the destructive accent.
        self._remove_widget = self._build_link(
            "✕", self.tr("Remove all"), "#990000"
        )
        self._remove_widget.mouseReleaseEvent = (  # type: ignore[assignment]
            lambda _e: self.removeAllClicked.emit()
        )
        h.addWidget(self._remove_widget)

        h.addStretch(1)

    @staticmethod
    def _build_link(glyph: str, text: str, color: str) -> QFrame:
        wrap = QFrame()
        wrap.setProperty("class", "row")
        wrap.setCursor(Qt.CursorShape.PointingHandCursor)
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(8)
        glyph_lbl = QLabel(glyph)
        glyph_lbl.setStyleSheet(
            f"color: {color}; font-weight: 600; font-size: 14pt;"
        )
        wl.addWidget(glyph_lbl)
        text_lbl = QLabel(text)
        text_lbl.setStyleSheet(f"color: {color}; font-weight: 500;")
        wl.addWidget(text_lbl)
        return wrap

    def set_remove_enabled(self, enabled: bool) -> None:
        # Hide the remove-all link entirely when there are no rows
        # to remove. Keeps the row from showing a no-op control.
        self._remove_widget.setVisible(enabled)


class SpeakerListView(QFrame):
    """Editable list of speakers, backed by `SpeakerTableModel`."""

    addRequested = pyqtSignal()

    def __init__(
        self, model: SpeakerTableModel, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._model = model
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._card = CardFrame(self)
        layout.addWidget(self._card)

        # The "+ Add speaker..." row is kept stable across rebuilds:
        # if it were re-created every time a speaker is added, the
        # nested QFileDialog / QInputDialog event loops would let the
        # rebuild fire and destroy the very row whose mouseReleaseEvent
        # is still on the stack -> "wrapped C/C++ object has been
        # deleted".
        self._add_row = _AddSpeakerRow()
        self._add_row.addClicked.connect(self.addRequested.emit)
        self._add_row.removeAllClicked.connect(self._remove_all)

        # Coalesce multiple model signals fired during a single
        # operation (e.g. rowsInserted + dataChanged) into one
        # rebuild on the next event-loop tick.
        self._rebuild_pending = False
        for sig in (
            self._model.dataChanged,
            self._model.rowsInserted,
            self._model.rowsRemoved,
            self._model.modelReset,
            self._model.layoutChanged,
        ):
            sig.connect(self._schedule_rebuild)

        self._rebuild()

    # ----- view rebuild --------------------------------------------------
    def _schedule_rebuild(self, *_args) -> None:
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(0, self._do_rebuild)

    def _do_rebuild(self) -> None:
        self._rebuild_pending = False
        self._rebuild()

    def _rebuild(self) -> None:
        # Detach the stable `+ Add speaker...` row first so it
        # survives the clear (we never deleteLater this instance,
        # because its mouseReleaseEvent may still be on the call
        # stack from the click that triggered the rebuild).
        layout = self._card._inner  # type: ignore[attr-defined]
        if self._add_row.parent() is not None:
            self._add_row.setParent(None)

        # `setParent(None)` detaches widgets from the visible tree
        # immediately; deleteLater then schedules destruction.
        # Without setParent(None) the old rows would keep painting
        # until Qt got around to deleting them.
        while layout.count() > 0:
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._add_row:
                w.setParent(None)
                w.deleteLater()

        rows = self._model.rows()
        for i, row in enumerate(rows):
            self._card.add_row(
                _SpeakerRow(
                    i,
                    row["speaker"],
                    row["file_path"],
                    on_edit=self._edit,
                    on_remove=self._remove,
                )
            )
        # The "Remove all" link only makes sense when there's
        # something to remove.
        self._add_row.set_remove_enabled(bool(rows))
        self._card.add_row(self._add_row)

    # ----- row actions ---------------------------------------------------
    def _edit(self, index: int) -> None:
        rows = self._model.rows()
        if not (0 <= index < len(rows)):
            return
        current = rows[index]["speaker"]
        new_name, ok = QInputDialog.getText(
            self,
            self.tr("Rename speaker"),
            self.tr("New name:"),
            text=current,
        )
        if ok and new_name.strip():
            model_index = self._model.index(index, 0)
            self._model.setData(model_index, new_name.strip())

    def _remove(self, index: int) -> None:
        self._model.remove_row(index)

    def _remove_all(self) -> None:
        """Wipe every row from the speaker model. Called from the
        in-card 'Remove all' link."""
        self._model.set_rows([])

    def remove_all(self) -> None:
        """Public wrapper so the 'New project' button on the Run
        page can also clear the speaker list."""
        self._remove_all()

    # ----- drag-drop -----------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        paths = mime_audio_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        paths = mime_audio_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = mime_audio_paths(event.mimeData())
        if not paths:
            super().dropEvent(event)
            return
        for path in paths:
            default_name = os.path.splitext(os.path.basename(path))[0]
            speaker, ok = QInputDialog.getText(
                self,
                self.tr("Speaker name"),
                self.tr("Speaker name for {0}:").format(os.path.basename(path)),
                text=default_name,
            )
            if ok and speaker.strip():
                self._model.add_row(speaker.strip(), path)
        event.acceptProposedAction()
